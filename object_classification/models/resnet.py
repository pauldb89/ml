from dataclasses import dataclass
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

from object_classification.data import Batch


@dataclass
class ConvConfig:
    channels: int
    kernel_size: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]


@dataclass
class ConvLayerConfig:
    block_config: Sequence[ConvConfig]
    num_blocks: int


RESNET_50_CONFIG = (
    ConvLayerConfig(
        block_config=(
            ConvConfig(kernel_size=1, channels=64, padding=0),
            ConvConfig(kernel_size=3, channels=64, padding=1),
            ConvConfig(kernel_size=1, channels=256, padding=0),
        ),
        num_blocks=3,
    ),
    ConvLayerConfig(
        block_config=(
            ConvConfig(kernel_size=1, channels=128, padding=0),
            ConvConfig(kernel_size=3, channels=128, padding=1),
            ConvConfig(kernel_size=1, channels=512, padding=0),
        ),
        num_blocks=4,
    ),
    ConvLayerConfig(
        block_config=(
            ConvConfig(kernel_size=1, channels=256, padding=0),
            ConvConfig(kernel_size=3, channels=256, padding=1),
            ConvConfig(kernel_size=1, channels=1024, padding=0),
        ),
        num_blocks=6,
    ),
    ConvLayerConfig(
        block_config=(
            ConvConfig(kernel_size=1, channels=512, padding=0),
            ConvConfig(kernel_size=3, channels=512, padding=1),
            ConvConfig(kernel_size=1, channels=2048, padding=0),
        ),
        num_blocks=3,
    ),
)


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int,
        padding: Union[int, Tuple[int, int]],
    ):
        super(Conv2D, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, block_config: Sequence[ConvConfig], in_channels: int, downsample: bool):
        super().__init__()

        conv_layers = [
            Conv2D(
                in_channels=in_channels,
                out_channels=block_config[0].channels,
                kernel_size=block_config[0].kernel_size,
                stride=2 if downsample else 1,
                padding=block_config[0].padding,
            )
        ]
        channels = block_config[0].channels

        for config in block_config[1:]:
            conv_layers.extend([
                nn.ReLU(),
                Conv2D(
                    in_channels=channels,
                    out_channels=config.channels,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=config.padding,
                )
            ])

        self.conv_layers = nn.Sequential(*conv_layers)

        out_channels = block_config[-1].channels
        if in_channels == out_channels:
            self.residual_transform = nn.Identity()
        else:
            self.residual_transform = Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2 if downsample else 1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_layers(x) + self.residual_transform(x))


class ConvLayer(nn.Module):
    def __init__(self, config: ConvLayerConfig, in_channels: int, downsample: bool):
        super().__init__()

        layers = []
        for i in range(config.num_blocks):
            layers.append(
                ConvBlock(
                    block_config=config.block_config,
                    in_channels=in_channels,
                    downsample=downsample and i == 0,
                ),
            )
            in_channels = config.block_config[-1].channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNet(nn.Module):
    def __init__(self, config: Sequence[ConvLayerConfig] = RESNET_50_CONFIG, num_classes: int = 1000):
        super().__init__()

        in_channels = 64
        conv_blocks = []
        for i, conv_layer_config in enumerate(config):
            conv_blocks.append(ConvLayer(conv_layer_config, in_channels=in_channels, downsample=i > 0))
            in_channels = conv_layer_config.block_config[-1].channels

        self.layers = nn.Sequential(
            Conv2D(kernel_size=7, in_channels=3, out_channels=64, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *conv_blocks,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        batch = batch.to(torch.device("cuda"))
        logits = self.layers(batch.images)
        return F.cross_entropy(logits, batch.classes)

    def eval_forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.layers(images), dim=1)
