import enum
import math
from typing import Dict
from typing import FrozenSet
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class NoiseSchedule(nn.Module):
	def get_alphas(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		return self.alphas[diffusion_steps].view(-1, 1, 1, 1)

	def get_alpha_bars(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		return self.alpha_bars[diffusion_steps].view(-1, 1, 1, 1)

	def get_alpha_bars_prev(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		return self.alpha_bars_prev[diffusion_steps].view(-1, 1, 1, 1)


class LinearNoiseSchedule(NoiseSchedule):
	"""
	Linear diffusion noise schedule used in https://arxiv.org/pdf/2006.11239v2.pdf.
	"""
	def __init__(self, beta_start: float, beta_end: float, num_diffusion_steps: int):
		super().__init__()

		self.beta_start = beta_start * (1000 / num_diffusion_steps)
		self.beta_end = beta_end * (1000 / num_diffusion_steps)
		self.num_diffusion_steps = num_diffusion_steps

		alpha_bar = 1
		alpha_bars = []
		alphas = []
		for step in range(num_diffusion_steps):
			beta = beta_start + step / (num_diffusion_steps - 1) * (beta_end - beta_start)
			alpha = 1 - beta
			alphas.append(alpha)
			alpha_bar *= alpha
			alpha_bars.append(alpha_bar)

		self.register_buffer("alphas", torch.tensor(alphas))
		self.register_buffer("alpha_bars", torch.tensor(alpha_bars))
		self.register_buffer("alpha_bars_prev", torch.tensor([1.0] + alpha_bars[:-1]))

	def __repr__(self) -> str:
		return (
			f"LinearNoiseSchedule("
			f"beta_start={self.beta_start}, "
			f"beta_end={self.beta_end}, "
			f"num_diffusion_steps={self.num_diffusion_steps}"
			f")"
		)


class CosineSquaredNoiseSchedule(NoiseSchedule):
	"""
	Improved diffusion noise schedule proposed in https://arxiv.org/abs/2102.09672.
	"""
	def __init__(self, num_diffusion_steps: int, smooth_factor: float = 0.008):
		super().__init__()

		self.smooth_factor = smooth_factor
		self.num_diffusion_steps = num_diffusion_steps

		def f(t: int) -> float:
			return math.cos((t / num_diffusion_steps + smooth_factor) / (1 + smooth_factor) * 2 * math.pi) ** 2

		alpha_bars = []
		for step in range(num_diffusion_steps):
			alpha_bar = f(step + 1) / f(0)
			alpha_bars.append(alpha_bar)

		self.register_buffer("alpha_bars", torch.tensor(alpha_bars))
		self.register_buffer("alpha_bars_prev", torch.tensor([1.0] + alpha_bars[:-1]))
		self.register_buffer("alphas", self.alpha_bars / self.alpha_bars_prev)

	def __repr__(self) -> str:
		return (
			f"CosineSquaredNoiseSchedule("
			f"num_diffusion_steps={self.num_diffusion_steps}, "
			f"smooth_factor={self.smooth_factor}"
			f")"
		)


class TimeEmbedder(nn.Module):
	"""
	Generate sine / cosine positional (time) embeddings based on https://arxiv.org/pdf/1706.03762.pdf.
	"""
	def __init__(self, embed_dim: int, scale: float = 10_000):
		super().__init__()

		assert embed_dim % 2 == 0

		self.scale = scale
		self.embed_dim = embed_dim

	def forward(self, diffusion_steps: torch.Tensor) -> torch.Tensor:

		num_frequencies = self.embed_dim // 2
		exp_scale = math.log(self.scale) / (num_frequencies - 1)
		# Pick sine / cosine frequencies with exponential decay between [1, 1 / self.scale].
		frequencies = torch.exp(torch.arange(num_frequencies, device="cuda") * -exp_scale)
		raw_embedding = torch.outer(diffusion_steps, frequencies)
		return torch.cat([torch.sin(raw_embedding), torch.cos(raw_embedding)], dim=1)

	def __repr__(self) -> str:
		return f"TimeEmbedder(scale={self.scale}, embed_dim={self.embed_dim})"


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, groups: int):
		super().__init__()

		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

	def forward(self, x: torch.Tensor, time_scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.norm(self.conv(x))

		if time_scale_shift is not None:
			batch_size, channels = time_scale_shift.size()
			scale, shift = torch.chunk(time_scale_shift.view(batch_size, channels, 1, 1), chunks=2, dim=1)
			x = x * (scale + 1) + shift

		return F.silu(x)


class ResnetBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, groups: int = 8):
		super().__init__()

		self.block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, groups=groups)
		self.block2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, groups=groups)

		# https://arxiv.org/pdf/2006.11239v2.pdf adds linear(silu(time_embedding)) in each residual block. Inspired by
		# `denoising-diffusion-pytorch` we instead learn two vectors, scale and shift, to transform
		# silu(time_embedding) as scale * silu(time_embedding) + shift.
		self.time_mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(in_features=time_embed_dim, out_features=2 * out_channels)
		)

		if in_channels != out_channels:
			self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
		else:
			self.residual = nn.Identity()

	def forward(self, input_tensor: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
		time_scale_shift = self.time_mlp(F.silu(time_embedding))

		x = self.block1(input_tensor, time_scale_shift=time_scale_shift)
		x = self.block2(x, time_scale_shift=None)

		return self.residual(input_tensor) + x


class AttentionBlock(nn.Module):
	def __init__(self, embedding_dim: int, num_heads: int = 4):
		super().__init__()
		self.pre_norm = nn.LayerNorm(embedding_dim)
		self.block = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embedding_dim, batch_first=True)

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		batch_size, num_channels, height, width = input_tensor.size()
		x = input_tensor.view(batch_size, num_channels, -1).permute(0, 2, 1)
		x = self.pre_norm(x)
		x, _ = self.block(query=x, key=x, value=x)
		x = x.permute(0, 2, 1).view(batch_size, num_channels, height, width)
		return input_tensor + x


class ResolutionMode(enum.IntEnum):
	NONE = 1
	UPSAMPLE = 2
	DOWNSAMPLE = 3


class DenoisingBlock(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		time_embed_dim: int,
		num_blocks_per_scale: int,
		use_attention: bool,
		resolution_scale_mode: ResolutionMode,
	):
		super().__init__()

		self.resnet_blocks = nn.ModuleList(
			ResnetBlock(
				in_channels=in_channels,
				out_channels=in_channels,
				time_embed_dim=time_embed_dim,
			)
			for _ in range(num_blocks_per_scale)
		)

		self.attention_block = AttentionBlock(embedding_dim=in_channels) if use_attention else nn.Identity()

		if resolution_scale_mode == ResolutionMode.NONE:
			self.resolution_block = nn.Conv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=3,
				padding=1,
			)
		elif resolution_scale_mode == ResolutionMode.DOWNSAMPLE:
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

	def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		for resnet_block in self.resnet_blocks:
			x = resnet_block(input_tensor=x, time_embedding=time_embedding)

		x = self.attention_block(x)

		return self.resolution_block(x), x


class DenoisingModel(nn.Module):
	def __init__(
		self,
		initial_channels: int,
		num_blocks_per_scale: int,
		scale_channel_multipliers: Tuple[int, ...],
		attention_block_ids: FrozenSet[int],
		time_embed_dim: int,
	):
		super().__init__()

		self.num_scales = len(scale_channel_multipliers)

		self.initial_channels = initial_channels

		self.out_channels = [self.initial_channels * multiplier for multiplier in scale_channel_multipliers]
		self.in_channels = [self.initial_channels] + self.out_channels[:-1]

		self.time_embedder_mlp = nn.Sequential(
			TimeEmbedder(embed_dim=time_embed_dim),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
		)
		self.init_conv = nn.Conv2d(in_channels=3, out_channels=initial_channels, kernel_size=3, padding=1)

		self.encoder_layers = nn.ModuleList()
		self.decoder_layers = nn.ModuleList()
		for block_id in range(self.num_scales):
			is_last = block_id + 1 == self.num_scales
			self.encoder_layers.append(
				DenoisingBlock(
					in_channels=self.in_channels[block_id],
					out_channels=self.out_channels[block_id],
					time_embed_dim=time_embed_dim,
					num_blocks_per_scale=num_blocks_per_scale,
					resolution_scale_mode=ResolutionMode.NONE if is_last else ResolutionMode.DOWNSAMPLE,
					use_attention=block_id in attention_block_ids,
				)
			)

		self.mid_processor = DenoisingBlock(
			in_channels=self.out_channels[-1],
			out_channels=self.out_channels[-1],
			time_embed_dim=time_embed_dim,
			num_blocks_per_scale=num_blocks_per_scale,
			use_attention=True,
			resolution_scale_mode=ResolutionMode.NONE,
		)

		for block_id in reversed(range(self.num_scales)):
			is_last = block_id == 0
			self.decoder_layers.append(
				DenoisingBlock(
					# I wonder if this could be simplified to just feed the downsampled state.
					in_channels=self.in_channels[block_id] + self.out_channels[block_id],
					out_channels=self.in_channels[block_id],
					time_embed_dim=time_embed_dim,
					num_blocks_per_scale=num_blocks_per_scale,
					resolution_scale_mode=ResolutionMode.NONE if is_last else ResolutionMode.UPSAMPLE,
					use_attention=block_id in attention_block_ids,
				)
			)

		self.out_conv = nn.Conv2d(in_channels=initial_channels, out_channels=3, kernel_size=1)

	def forward(self, noised_images: torch.Tensor, diffusion_steps: torch.Tensor) -> torch.Tensor:
		time_embedding = self.time_embedder_mlp(diffusion_steps)

		x = self.init_conv(noised_images)

		encoder_states = []
		for block_id, encoder_layer in enumerate(self.encoder_layers):
			x, encoder_state = encoder_layer(x, time_embedding)
			encoder_states.append(encoder_state)

		x, _ = self.mid_processor(x, time_embedding)

		for block_id, decoder_layer in enumerate(self.decoder_layers):
			x = torch.cat([x, encoder_states[self.num_scales - block_id - 1]], dim=1)
			x, _ = decoder_layer(x, time_embedding)

		return self.out_conv(x)


class DiffusionModel(nn.Module):
	def __init__(
		self,
		num_diffusion_steps: int,
		noise_schedule: NoiseSchedule,
		initial_channels: int = 128,
		num_blocks_per_scale: int = 2,
		scale_channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4, 4),
		attention_block_ids: FrozenSet[int] = frozenset([3]),
		resolution: int = 256,
	):
		super().__init__()

		self.num_diffusion_steps = num_diffusion_steps
		self.resolution = resolution
		self.noise_schedule = noise_schedule

		self.register_buffer("steps", torch.tensor(0))
		self.model = DenoisingModel(
			initial_channels=initial_channels,
			num_blocks_per_scale=num_blocks_per_scale,
			scale_channel_multipliers=scale_channel_multipliers,
			attention_block_ids=attention_block_ids,
			time_embed_dim=initial_channels * 4,
		)

	def apply_noise(self, images: torch.Tensor, diffusion_steps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
		"""
		Computes x_t (image after t steps in the forward diffusion process), given a batch of images (x_0), a batch of
		diffusion steps t and sampled normal gaussian noise (epsilon).

		The formula for the forward process is x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) epsilon.
		Equation (4) from https://arxiv.org/pdf/2006.11239v2.pdf.
		"""
		assert images.size() == noise.size()

		alpha_bars = self.noise_schedule.get_alpha_bars(diffusion_steps)
		return torch.sqrt(alpha_bars) * images + torch.sqrt(1 - alpha_bars) * noise

	def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
		"""
		This implements the training loop for diffusion models (DDPM). For more details, see Algorithm 1 from
		https://arxiv.org/pdf/2006.11239v2.pdf.
		"""
		# Scale images to [-1, 1].
		images = 2 * images.cuda() - 1

		batch_size = images.size(0)
		diffusion_steps = torch.randint(low=0, high=self.num_diffusion_steps-1, size=(batch_size,), device="cuda")

		noise = torch.randn_like(images)

		noised_images = self.apply_noise(images=images, diffusion_steps=diffusion_steps, noise=noise)

		inferred_noise = self.model(noised_images=noised_images, diffusion_steps=diffusion_steps)

		return {
			"loss": F.mse_loss(noise, inferred_noise)
		}

	def reverse_process_mean(
		self,
		images: torch.Tensor,
		diffusion_steps: torch.Tensor,
		inferred_noise: torch.Tensor,
	) -> torch.Tensor:
		"""
		Computes the posterior mean of the images at the previous diffusion step (e.g. t-1) given the images at
		step t and the inferred noise going from t -> t-1.

		We can compute the mean in two ways:
		1. mu_(x_t, t) = 1 / sqrt(alpha_t) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) eps(x_t, t)).
		Equation (11) from https://arxiv.org/pdf/2006.11239v2.pdf; we replace beta_t = (1 - alpha_t).
		2. x_0(x_t, t) = 1 / sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t) * eps(x_t, t)
		x_0(x_t, t) <- clip(x_0(x_t, t), -1, 1).
		mu_t(x_t, x_0) = sqrt(alpha_bar_{t-1}) (1 - alpha_t) / (1 - alpha_bar_t) x_0
			+ sqrt(alpha_t) (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) x_t
		The first formula can be derived from Equation (4) and the second is Equation (7) from
		https://arxiv.org/pdf/2006.11239v2.pdf. More details regarding the derivations can be found at
		https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

		Note: The second method permits clipping of the projected x_0 which is found to be necessary (empirically) for
		high quality samples. Some discussion on the fact that this is necessary and some suggestions for further
		refining the technique can be found in Section 2.3 of https://arxiv.org/pdf/2205.11487.pdf.
		"""
		alphas = self.noise_schedule.get_alphas(diffusion_steps)
		alpha_bars_t = self.noise_schedule.get_alpha_bars(diffusion_steps)
		alpha_bars_prev_t = self.noise_schedule.get_alpha_bars_prev(diffusion_steps)
		# return (images - (1 - alphas) / torch.sqrt(1 - alpha_bars) * inferred_noise) / torch.sqrt(alphas)
		start_images = (images - torch.sqrt(1 - alpha_bars_t) * inferred_noise) / torch.sqrt(alpha_bars_t)
		start_images = torch.clip(start_images, min=-1, max=1)
		return (
			torch.sqrt(alpha_bars_prev_t) * (1 - alphas) * start_images +
			torch.sqrt(alphas) * (1 - alpha_bars_prev_t) * images
		) / (1 - alpha_bars_t)

	def reverse_process_std(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		"""
		Return the reverse process standard deviation.

		The formula for the variance is:
		sigma_t^2 = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * (1 - alpha_t).

		For more details, see the first few lines of Section 3.2 of https://arxiv.org/pdf/2006.11239v2.pdf.

		Note that this formula doesn't work for step 0 (since we require the alpha_bar at step t-1). This is not an
		issue because we return the mean without adding any noise at step 0.
		"""
		alphas = self.noise_schedule.get_alphas(diffusion_steps)
		alpha_bars_t = self.noise_schedule.get_alpha_bars(diffusion_steps)
		alpha_bars_prev_t = self.noise_schedule.get_alpha_bars_prev(diffusion_steps)
		return torch.sqrt((1 - alpha_bars_prev_t) * (1 - alphas) / (1 - alpha_bars_t))

	def sample(self, batch_size: int) -> torch.Tensor:
		"""
		This implements the sampling loop for diffusion models. For more details, see Algorithm 2 from
		https://arxiv.org/pdf/2006.11239v2.pdf.
		"""
		images = torch.randn(batch_size, 3, self.resolution, self.resolution, device="cuda")
		for diffusion_step in reversed(range(self.num_diffusion_steps)):
			diffusion_steps = torch.full(size=(batch_size,), fill_value=diffusion_step, device="cuda")
			inferred_noise = self.model(images, diffusion_steps=diffusion_steps)

			mean = self.reverse_process_mean(
				images=images,
				diffusion_steps=diffusion_steps,
				inferred_noise=inferred_noise,
			)

			if diffusion_step > 0:
				std = self.reverse_process_std(diffusion_steps=diffusion_steps)
				images = mean + torch.randn_like(images) * std
			else:
				images = mean

		return images / 2 + 0.5


class AverageModelWrapper(torch.optim.swa_utils.AveragedModel):
	def sample(self, *args, **kwargs) -> torch.Tensor:
		return self.module.sample(*args, **kwargs)
