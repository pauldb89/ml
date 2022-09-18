import random
from typing import Dict
from typing import FrozenSet
from typing import Tuple

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.cuda.amp import autocast
from torch.optim.swa_utils import AveragedModel
from transformers import AutoConfig
from transformers import CLIPTextModel
from transformers import T5EncoderModel

from diffusion.common.modules import ConvBlock
from diffusion.common.modules import ResolutionMode
from diffusion.common.modules import SelfAttentionBlock
from diffusion.common.modules import TimeEmbedder
from diffusion.common.noise_schedule import NoiseSchedule
from diffusion.conditional.data import Batch


class ConditionalResnetBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, text_embed_dim: int):
		super().__init__()

		self.time_mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(time_embed_dim, 2 * out_channels),
		)
		self.text_mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(text_embed_dim, 2 * out_channels),
		)
		self.block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels)
		self.block2 = ConvBlock(in_channels=out_channels, out_channels=out_channels)
		self.block3 = ConvBlock(in_channels=out_channels, out_channels=out_channels)

		if in_channels == out_channels:
			self.residual = nn.Identity()
		else:
			self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

	def forward(
		self,
		x: torch.Tensor,
		time_embedding: torch.Tensor,
		pooled_text_embedding: torch.Tensor,
	) -> torch.Tensor:
		time_scale_shift = self.time_mlp(time_embedding)
		text_scale_shift = self.text_mlp(pooled_text_embedding)

		x = self.block1(x, scale_shift=time_scale_shift)
		x = self.block2(x, scale_shift=text_scale_shift)
		x = self.block3(x, scale_shift=None)

		return self.residual(x) + x


class CrossAttentionBlock(nn.Module):
	def __init__(
		self,
		embed_dim: int,
		time_embed_dim: int,
		text_embed_dim: int,
		num_heads: int = 4,
	):
		super().__init__()

		self.time_mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(time_embed_dim, embed_dim),
			nn.LayerNorm(embed_dim),
		)
		self.text_mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(text_embed_dim, embed_dim),
			nn.LayerNorm(embed_dim),
		)

		self.pre_norm = nn.LayerNorm(embed_dim)
		self.block = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

	def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
		time_embedding = torch.unsqueeze(self.time_mlp(time_embedding), dim=1)
		text_embeddings = self.text_mlp(text_embeddings)
		context = torch.cat([time_embedding, text_embeddings], dim=1)

		batch_size, channels, height, width = x.size()

		queries = x.view(batch_size, channels, height * width).permute(dims=[0, 2, 1])
		queries = self.pre_norm(queries)
		output, _ = self.block(query=queries, key=context, value=context)
		output = output.permute(dims=[0, 2, 1]).view(batch_size, channels, height, width)

		return x + output


class ConditionalDenoisingBlock(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		time_embed_dim: int,
		text_embed_dim: int,
		num_blocks_per_scale: int,
		use_self_attention: bool,
		use_cross_attention: bool,
		resolution_mode: ResolutionMode,
	):
		super().__init__()

		self.resnet_blocks = nn.ModuleList([
			ConditionalResnetBlock(
				in_channels=in_channels,
				out_channels=in_channels,
				time_embed_dim=time_embed_dim,
				text_embed_dim=text_embed_dim,
			)
			for _ in range(num_blocks_per_scale)
		])

		self.self_attention_block = SelfAttentionBlock(embed_dim=in_channels) if use_self_attention else None
		if use_cross_attention:
			self.cross_attention_block = CrossAttentionBlock(
				embed_dim=in_channels,
				time_embed_dim=time_embed_dim,
				text_embed_dim=text_embed_dim,
			)
		else:
			self.cross_attention_block = None

		if resolution_mode == ResolutionMode.NONE:
			self.resolution_block = nn.Conv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=3,
				padding=1,
			)
		elif resolution_mode == ResolutionMode.DOWNSAMPLE:
			self.resolution_block = nn.Conv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=4,
				stride=2,
				padding=1,
			)
		else:
			self.resolution_block = nn.Sequential(
				nn.UpsamplingNearest2d(scale_factor=2.0),
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
			)

	def forward(
		self,
		x: torch.Tensor,
		time_embedding: torch.Tensor,
		text_embeddings: torch.Tensor,
		pooled_text_embedding: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		for resnet_block in self.resnet_blocks:
			x = resnet_block(x, time_embedding=time_embedding, pooled_text_embedding=pooled_text_embedding)

		if self.self_attention_block is not None:
			x = self.self_attention_block(x)

		if self.cross_attention_block is not None:
			x = self.cross_attention_block(x, time_embedding=time_embedding, text_embeddings=text_embeddings)

		return self.resolution_block(x), x


class ConditionalDenoisingModel(nn.Module):
	def __init__(
		self,
		initial_channels: int,
		scale_channel_multipliers: Tuple[int, ...],
		time_embed_dim: int,
		text_embed_dim: int,
		num_blocks_per_scale: int,
		self_attention_block_ids: FrozenSet[int],
		cross_attention_block_ids: FrozenSet[int],
	):
		super().__init__()

		self.initial_channels = initial_channels
		self.time_embed_dim = time_embed_dim

		self.time_mlp = nn.Sequential(
			TimeEmbedder(time_embed_dim),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
		)

		# In https://github.com/lucidrains/imagen-pytorch, much larger kernel sizes are used, including an option
		# to allocate different channels to different kernel sizes by concatenating feature maps resulting from several
		# convolutional layers.
		self.conv_in = nn.Conv2d(in_channels=3, out_channels=initial_channels, kernel_size=3, padding=1)
		self.conv_out = nn.Conv2d(in_channels=initial_channels, out_channels=3, kernel_size=3, padding=1)
		
		self.num_scales = len(scale_channel_multipliers)
		self.out_channels = [scale * initial_channels for scale in scale_channel_multipliers]
		self.in_channels = [initial_channels] + self.out_channels[:-1]

		self.encoder_layers = nn.ModuleList()
		for block_idx in range(self.num_scales):
			is_last = block_idx == self.num_scales - 1
			self.encoder_layers.append(
				ConditionalDenoisingBlock(
					in_channels=self.in_channels[block_idx],
					out_channels=self.out_channels[block_idx],
					time_embed_dim=time_embed_dim,
					text_embed_dim=text_embed_dim,
					num_blocks_per_scale=num_blocks_per_scale,
					use_self_attention=block_idx in self_attention_block_ids,
					use_cross_attention=block_idx in cross_attention_block_ids,
					resolution_mode=ResolutionMode.NONE if is_last else ResolutionMode.DOWNSAMPLE,
				)
			)

		self.mid_processor = ConditionalDenoisingBlock(
			in_channels=self.in_channels[-1],
			out_channels=self.out_channels[-1],
			time_embed_dim=time_embed_dim,
			text_embed_dim=text_embed_dim,
			num_blocks_per_scale=num_blocks_per_scale,
			use_self_attention=True,
			use_cross_attention=True,
			resolution_mode=ResolutionMode.NONE,
		)

		self.decoder_layers = nn.ModuleList()
		for block_idx in reversed(range(self.num_scales)):
			is_last = block_idx == 0
			self.decoder_layers.append(
				ConditionalDenoisingBlock(
					in_channels=self.in_channels[block_idx] + self.out_channels[block_idx],
					out_channels=self.in_channels[block_idx],
					time_embed_dim=time_embed_dim,
					text_embed_dim=text_embed_dim,
					num_blocks_per_scale=num_blocks_per_scale,
					use_self_attention=block_idx in self_attention_block_ids,
					use_cross_attention=block_idx in cross_attention_block_ids,
					resolution_mode=ResolutionMode.NONE if is_last else ResolutionMode.UPSAMPLE,
				)
			)

	def forward(
		self,
		corrupted_images: torch.Tensor,
		diffusion_steps: torch.Tensor,
		text_embeddings: torch.Tensor,
	) -> torch.Tensor:
		time_embedding = self.time_mlp(diffusion_steps)
		pooled_text_embedding = torch.mean(text_embeddings, dim=1)

		x = self.conv_in(corrupted_images)

		encoded_states = []
		for encoder_layer in self.encoder_layers:
			x, encoded_state = encoder_layer(
				x=x,
				time_embedding=time_embedding,
				text_embeddings=text_embeddings,
				pooled_text_embedding=pooled_text_embedding,
			)
			encoded_states.append(encoded_state)

		x, _ = self.mid_processor(
			x=x,
			time_embedding=time_embedding,
			text_embeddings=text_embeddings,
			pooled_text_embedding=pooled_text_embedding,
		)

		for block_idx, decoder_layer in enumerate(self.decoder_layers):
			state = torch.cat([x, encoded_states[self.num_scales - block_idx - 1]], dim=1)
			x, _ = decoder_layer(
				x=state,
				time_embedding=time_embedding,
				text_embeddings=text_embeddings,
				pooled_text_embedding=pooled_text_embedding,
			)

		return self.conv_out(x)


class ConditionalDiffusionModel(nn.Module):
	def __init__(
		self,
		text_encoder: str,
		resolution: int,
		num_diffusion_steps: int,
		noise_schedule: NoiseSchedule,
		initial_channels: int = 128,
		num_blocks_per_scale: int = 1,
		scale_channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4, 4),
		self_attention_block_ids: FrozenSet[int] = frozenset([3, 4, 5]),
		cross_attention_block_ids: FrozenSet[int] = frozenset([3, 4, 5]),
		conditional_dropout: float = 0.1,
	):
		super().__init__()

		_, short_name = text_encoder.split("/")
		text_encoder_class = CLIPTextModel if short_name.startswith("clip") else T5EncoderModel
		self.text_encoder = text_encoder_class.from_pretrained(text_encoder)

		text_encoder_config = AutoConfig.from_pretrained(text_encoder)
		if short_name.startswith("clip"):
			self.text_embed_dim = text_encoder_config.projection_dim
		else:
			self.text_embed_dim = text_encoder_config.d_model

		self.resolution = resolution
		self.num_diffusion_steps = num_diffusion_steps
		self.conditional_dropout = conditional_dropout

		self.noise_schedule = noise_schedule

		self.model = ConditionalDenoisingModel(
			initial_channels=initial_channels,
			scale_channel_multipliers=scale_channel_multipliers,
			time_embed_dim=initial_channels * 4,
			text_embed_dim=self.text_embed_dim,
			num_blocks_per_scale=num_blocks_per_scale,
			self_attention_block_ids=self_attention_block_ids,
			cross_attention_block_ids=cross_attention_block_ids,
		)

	def apply_noise(self, images: torch.Tensor, diffusion_steps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
		"""
		Computes x_t (image after t steps in the forward diffusion process), given a batch of images (x_0), a batch of
		diffusion steps t and sampled normal gaussian noise (epsilon).

		The formula for the forward process is x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) epsilon.
		Equation (4) from https://arxiv.org/pdf/2006.11239v2.pdf.

		Note that the equation is expressed in terms of variance, so we need to use the square root for noise
		multiplier. Full derivation at https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.
		"""
		alpha_bars = self.noise_schedule.get_alpha_bars(diffusion_steps)
		return torch.sqrt(alpha_bars) * images + torch.sqrt(1 - alpha_bars) * noise

	def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
		batch = batch.to(torch.device("cuda"))

		batch_size, sequence_length = batch.token_ids.size()

		images = 2 * batch.images - 1

		with torch.inference_mode():
			with autocast(enabled=False):
				if random.random() > self.conditional_dropout:
					text_embeddings = self.text_encoder(
						input_ids=batch.token_ids,
						attention_mask=batch.token_masks,
					).last_hidden_state
				else:
					text_embeddings = torch.zeros(batch_size, sequence_length, self.text_embed_dim, device="cuda")

		diffusion_steps = torch.randint(0, self.num_diffusion_steps-1, size=(batch_size, ), device="cuda")
		noise = torch.randn_like(images)

		corrupted_images = self.apply_noise(images=images, diffusion_steps=diffusion_steps, noise=noise)

		inferred_noise = self.model(
			corrupted_images=corrupted_images,
			diffusion_steps=diffusion_steps,
			text_embeddings=text_embeddings,
		)

		return {"loss": F.mse_loss(inferred_noise, noise)}

	def reverse_process_mean(
		self,
		images: torch.Tensor,
		diffusion_steps: torch.Tensor,
		inferred_noise: torch.Tensor,
	) -> torch.Tensor:
		alpha_bars_t = self.noise_schedule.get_alpha_bars(diffusion_steps)
		alpha_bars_prev_t = self.noise_schedule.get_alpha_bars_prev(diffusion_steps)
		alphas = self.noise_schedule.get_alphas(diffusion_steps)

		inferred_original_images = (images - torch.sqrt(1 - alpha_bars_t) * inferred_noise) / torch.sqrt(alpha_bars_t)
		inferred_original_images = torch.clamp(inferred_original_images, min=-1.0, max=1.0)

		return (
			torch.sqrt(alpha_bars_prev_t) * (1 - alphas) * inferred_original_images +
			torch.sqrt(alphas) * (1 - alpha_bars_prev_t) * images
		) / (1 - alpha_bars_t)

	def reverse_process_std(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		alpha_bars_t = self.noise_schedule.get_alpha_bars(diffusion_steps)
		alpha_bars_prev_t = self.noise_schedule.get_alpha_bars_prev(diffusion_steps)
		alphas = self.noise_schedule.get_alphas(diffusion_steps)
		return torch.sqrt((1 - alpha_bars_prev_t) * (1 - alphas) / (1 - alpha_bars_t))

	def sample(self, batch: Batch, guidance_strength: float = 0.5) -> torch.Tensor:
		batch = batch.to(torch.device("cuda"))

		images = torch.randn(len(batch), 3, self.resolution, self.resolution, device="cuda")
		with torch.inference_mode():
			with autocast(enabled=False):
				text_embeddings = self.text_encoder(
					input_ids=batch.token_ids,
					attention_mask=batch.token_masks,
				).last_hidden_state
				zero_embeddings = torch.zeros_like(text_embeddings)

		for diffusion_step in tqdm.tqdm(list(reversed(range(self.num_diffusion_steps)))):
			diffusion_steps = torch.full(size=(len(batch),), fill_value=diffusion_step, device="cuda")
			conditional_noise = self.model(
				corrupted_images=images,
				diffusion_steps=diffusion_steps,
				text_embeddings=text_embeddings,
			)

			unconditional_noise = self.model(
				corrupted_images=images,
				diffusion_steps=diffusion_steps,
				text_embeddings=zero_embeddings,
			)
			inferred_noise = (1 + guidance_strength) * conditional_noise - guidance_strength * unconditional_noise

			mean = self.reverse_process_mean(images, diffusion_steps=diffusion_steps, inferred_noise=inferred_noise)
			if diffusion_step > 0:
				std = self.reverse_process_std(diffusion_steps=diffusion_steps)
				images = mean + torch.randn_like(images) * std
			else:
				images = mean

		return images / 2 + 0.5


class AverageModelWrapper(AveragedModel):
	@property
	def resolution(self) -> int:
		return self.module.resolution

	def sample(self, *args, **kwargs) -> torch.Tensor:
		return self.module.sample(*args, **kwargs)
