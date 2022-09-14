import enum
import math
from typing import Optional

import torch
from torch import nn
from torch import nn
from torch import nn
from torch import nn
from torch.nn import functional as F


class TimeEmbedder(nn.Module):
	"""
	Generate sine / cosine positional (time) embeddings based on https://arxiv.org/pdf/1706.03762.pdf.
	"""
	def __init__(self, embed_dim: int, scale: float = 10_000):
		super().__init__()

		assert embed_dim % 2 == 0

		self.scale = scale
		self.embed_dim = embed_dim

		num_frequencies = self.embed_dim // 2
		# Pick sine / cosine frequencies with exponential decay between [1, 1 / self.scale].
		exp_scale = math.log(self.scale) / (num_frequencies - 1)
		self.register_buffer("frequencies", torch.exp(torch.arange(num_frequencies, device="cuda") * -exp_scale))

	def forward(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
		raw_embedding = torch.outer(diffusion_steps, self.frequencies)
		return torch.cat([torch.sin(raw_embedding), torch.cos(raw_embedding)], dim=1)

	def __repr__(self) -> str:
		return f"TimeEmbedder(scale={self.scale}, embed_dim={self.embed_dim})"


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
		super().__init__()

		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

	def forward(self, x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.norm(self.conv(x))

		if scale_shift is not None:
			batch_size, channels = scale_shift.size()
			scale, shift = torch.chunk(scale_shift.view(batch_size, channels, 1, 1), chunks=2, dim=1)
			x = x * (scale + 1) + shift

		return F.silu(x)


class ResolutionMode(enum.IntEnum):
	NONE = 1
	UPSAMPLE = 2
	DOWNSAMPLE = 3


class SelfAttentionBlock(nn.Module):
	def __init__(self, embed_dim: int, num_heads: int = 4):
		super().__init__()
		self.pre_norm = nn.LayerNorm(embed_dim)
		self.block = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, batch_first=True)

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		batch_size, num_channels, height, width = input_tensor.size()
		x = input_tensor.view(batch_size, num_channels, -1).permute(0, 2, 1)
		x = self.pre_norm(x)
		x, _ = self.block(query=x, key=x, value=x)
		x = x.permute(0, 2, 1).view(batch_size, num_channels, height, width)
		return input_tensor + x
