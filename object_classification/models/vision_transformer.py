import math

import torch
from torch import nn
from torch.nn import functional as F

from object_classification.models.vision_model import VisionModel


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, 3*hidden_dim, bias=False)
        self.scalar_norm = math.sqrt(hidden_dim / num_heads)

        torch.nn.init.xavier_uniform_(self.qkv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of size B x N x D.
        :return: A tensor of size B x N x D.
        """
        b, n, d = x.size()
        queries, keys, values = torch.split(self.qkv(x), split_size_or_sections=d, dim=-1)

        queries = torch.permute(queries.view(b, n, self.num_heads, -1), dims=(0, 2, 1, 3))
        keys = torch.permute(keys.view(b, n, self.num_heads, -1), dims=(0, 2, 1, 3))
        values = torch.permute(values.view(b, n, self.num_heads, -1), dims=(0, 2, 1, 3))

        attention = F.softmax(torch.einsum("bhik,bhjk->bhij", queries, keys) / self.scalar_norm, dim=-1)
        ret = torch.einsum("bhij,bhjk->bhik", attention, values)

        return torch.permute(ret, dims=(0, 2, 1, 3)).reshape(b, n, d)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super(TransformerBlock, self).__init__()

        self.attention_block = Residual(
            nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=1e-6),
                MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads),
            )
        )

        self.mlp_block = Residual(
            nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.Linear(hidden_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, hidden_dim)
            )
        )

        for module in self.mlp_block.module:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int, mlp_dim: int, num_heads: int):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(TransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, mlp_dim=mlp_dim))
        self.layers = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of size B x N x D of embeddings (e.g. projections of patches for ViT).
        :return: A B x N x D tensor representing the output embeddings for all inputs in the batch.
        """
        return self.norm(self.layers(x))


class TransformerAggregator(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        num_classes: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.projection = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        sequence_length = (image_size // patch_size) ** 2 + 1
        self.position_embeddings = nn.Parameter(torch.randn(1, sequence_length, hidden_dim) * 0.02)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.mlp_head = nn.Linear(hidden_dim, num_classes)

        # Init weights.
        fan_in = self.projection.in_channels * self.projection.kernel_size[0] * self.projection.kernel_size[1]
        nn.init.trunc_normal_(self.projection.weight, std=math.sqrt(1 / fan_in))

        torch.nn.init.zeros_(self.mlp_head.weight)
        torch.nn.init.zeros_(self.mlp_head.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]

        x = self.projection(images)
        x = x.reshape(batch_size, self.hidden_dim, -1).permute(dims=(0, 2, 1))

        x = torch.cat([self.class_token.repeat(batch_size, 1, 1), x], dim=1)

        x = x + self.position_embeddings
        x = self.encoder(x)

        x = self.mlp_head(torch.squeeze(x[:, 0, :], dim=1))
        return x


class VisionTransformer(VisionModel):
    def make_layers(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        num_classes: int,
    ) -> nn.Module:
        return TransformerAggregator(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_classes=num_classes,
        )
