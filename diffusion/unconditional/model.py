import random
from typing import Dict
from typing import FrozenSet
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from common.distributed import get_rank
from diffusion.common.modules import SelfAttentionBlock
from diffusion.common.modules import ConvBlock
from diffusion.common.modules import ResolutionMode

from diffusion.common.modules import TimeEmbedder
from diffusion.common.noise_schedule import NoiseSchedule


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

		x = self.block1(input_tensor, scale_shift=time_scale_shift)
		x = self.block2(x, scale_shift=None)

		return self.residual(input_tensor) + x


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

		self.attention_block = SelfAttentionBlock(embed_dim=in_channels) if use_attention else nn.Identity()

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
		self_conditioning: bool,
	):
		super().__init__()

		self.num_scales = len(scale_channel_multipliers)

		self.initial_channels = initial_channels
		self.self_conditioning = self_conditioning

		self.out_channels = [self.initial_channels * multiplier for multiplier in scale_channel_multipliers]
		self.in_channels = [self.initial_channels] + self.out_channels[:-1]

		self.time_embedder_mlp = nn.Sequential(
			TimeEmbedder(embed_dim=time_embed_dim),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
		)

		input_channels = 6 if self.self_conditioning else 3
		self.init_conv = nn.Conv2d(in_channels=input_channels, out_channels=initial_channels, kernel_size=3, padding=1)

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

	def forward(
		self,
		noisy_images: torch.Tensor,
		diffusion_steps: torch.Tensor,
		inferred_original_images: Optional[torch.Tensor],
	) -> torch.Tensor:
		time_embedding = self.time_embedder_mlp(diffusion_steps)

		if self.self_conditioning:
			if inferred_original_images is None:
				inferred_original_images = torch.zeros_like(noisy_images)
			x = torch.cat([noisy_images, inferred_original_images], dim=1)
		else:
			assert inferred_original_images is None
			x = noisy_images

		x = self.init_conv(x)

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
		resolution: int = 128,
		p2_loss_gamma: float = 0.0,
		p2_loss_k: float = 1.0,
		self_conditioning_rate: float = 0.0,
	):
		super().__init__()

		self.num_diffusion_steps = num_diffusion_steps
		self.resolution = resolution
		self.noise_schedule = noise_schedule

		self.p2_loss_gamma = p2_loss_gamma
		self.p2_loss_k = p2_loss_k
		self.self_conditioning_rate = self_conditioning_rate

		self.register_buffer("steps", torch.tensor(0))
		self.model = DenoisingModel(
			initial_channels=initial_channels,
			num_blocks_per_scale=num_blocks_per_scale,
			scale_channel_multipliers=scale_channel_multipliers,
			attention_block_ids=attention_block_ids,
			time_embed_dim=initial_channels * 4,
			self_conditioning=self.self_conditioning_rate > 0,
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
		assert images.size() == noise.size()

		alpha_bars = self.noise_schedule.get_alpha_bars(diffusion_steps)
		return torch.sqrt(alpha_bars) * images + torch.sqrt(1 - alpha_bars) * noise

	def infer_original_images(
		self,
		noisy_images: torch.Tensor,
		inferred_noise: torch.Tensor,
		diffusion_steps: torch.Tensor,
	) -> torch.Tensor:
		"""
		Compute expected original image based on a set of noisy images at time t and the inferred noised.
		Clip the resulting images to [-1, 1], useful for producing high quality samples.

		x_0(x_t, t) = 1 / sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t) * eps(x_t, t)
		x_0(x_t, t) <- clip(x_0(x_t, t), -1, 1).

		The formula can be derived from Equation (4) in https://arxiv.org/pdf/2006.11239v2.pdf. Some discussion on
		the importance of clipping can be found in Section 2.3 of https://arxiv.org/pdf/2205.11487.pdf.
		"""
		alpha_bars = self.noise_schedule.get_alpha_bars(diffusion_steps)
		original_image = (noisy_images - torch.sqrt(1 - alpha_bars) * inferred_noise) / torch.sqrt(alpha_bars)
		return torch.clip(original_image, min=-1, max=1.0)

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

		noisy_images = self.apply_noise(images=images, diffusion_steps=diffusion_steps, noise=noise)

		inferred_original_images = None
		if random.random() <= self.self_conditioning_rate:
			inferred_noise = self.model(
				noisy_images=noisy_images,
				diffusion_steps=diffusion_steps,
				inferred_original_images=None,
			)
			inferred_original_images = self.infer_original_images(
				noisy_images=noisy_images,
				inferred_noise=inferred_noise,
				diffusion_steps=diffusion_steps,
			).detach()

		inferred_noise = self.model(
			noisy_images=noisy_images,
			diffusion_steps=diffusion_steps,
			inferred_original_images=inferred_original_images,
		)

		mse_loss = torch.mean(F.mse_loss(noise, inferred_noise, reduction="none"), dim=[1, 2, 3])
		return {
			"loss": torch.mean(mse_loss * self.p2_loss_weight(diffusion_steps))
		}

	def p2_loss_weight(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		"""
		Implement perceptual prioritized loss weighting from https://arxiv.org/abs/2204.00227.

		Setting p2_loss_gamma=0.0 makes the loss weights 1.0 as in the original implementation proposed in
		https://arxiv.org/pdf/2006.11239.pdf.
		"""
		alpha_bars = self.noise_schedule.get_alpha_bars(diffusion_steps).view(-1)
		return (self.p2_loss_k + alpha_bars / (1 - alpha_bars)) ** -self.p2_loss_gamma

	def reverse_process_mean(
		self,
		images: torch.Tensor,
		diffusion_steps: torch.Tensor,
		inferred_noise: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
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
		original_images = self.infer_original_images(
			noisy_images=images,
			inferred_noise=inferred_noise,
			diffusion_steps=diffusion_steps,
		)
		mean = (
			torch.sqrt(alpha_bars_prev_t) * (1 - alphas) * original_images +
			torch.sqrt(alphas) * (1 - alpha_bars_prev_t) * images
		) / (1 - alpha_bars_t)
		return mean, original_images

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
		inferred_original_images = None

		for diffusion_step in reversed(range(self.num_diffusion_steps)):
			diffusion_steps = torch.full(size=(batch_size,), fill_value=diffusion_step, device="cuda")

			if self.self_conditioning_rate == 0:
				inferred_original_images = None

			inferred_noise = self.model(
				images,
				diffusion_steps=diffusion_steps,
				inferred_original_images=inferred_original_images,
			)

			mean, inferred_original_images = self.reverse_process_mean(
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
	@property
	def resolution(self) -> int:
		return self.module.resolution

	def sample(self, *args, **kwargs) -> torch.Tensor:
		return self.module.sample(*args, **kwargs)
