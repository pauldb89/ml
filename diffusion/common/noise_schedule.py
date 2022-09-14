import math

import torch
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
			return math.cos((t / num_diffusion_steps + smooth_factor) / (1 + smooth_factor) * math.pi / 2) ** 2

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
