from typing import Tuple
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

from object_classification.data import Batch


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        eps: float = 0.001,
    ):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels, eps=eps),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InceptionLayerBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = self.make_layers()

    def make_layers(self) -> nn.ModuleList:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x))
        return torch.cat(outputs, dim=1)


class InceptionLayerA(InceptionLayerBase):
    def __init__(self, in_channels: int, num_pool_branch_out_channels: int):
        self.in_channels = in_channels
        self.num_pool_branch_out_channels = num_pool_branch_out_channels

        super().__init__()

    def make_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ConvBlock(in_channels=self.in_channels, out_channels=64, kernel_size=1),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=48, kernel_size=1),
                # Note(pauldb): According to https://pytorch.org/assets/images/inception_v3.png, I think this should
                # kernel_size=3.
                ConvBlock(in_channels=48, out_channels=64, kernel_size=5, padding=2)
            ),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=64, kernel_size=1),
                ConvBlock(in_channels=64, out_channels=96, kernel_size=3, padding=1),
                ConvBlock(in_channels=96, out_channels=96, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels=self.in_channels, out_channels=self.num_pool_branch_out_channels, kernel_size=1)
            ),
        ])


class InceptionLayerB(InceptionLayerBase):
    def __init__(self, in_channels: int):
        self.in_channels = in_channels

        super().__init__()

    def make_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ConvBlock(in_channels=self.in_channels, out_channels=384, kernel_size=3, stride=2),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=64, kernel_size=1),
                ConvBlock(in_channels=64, out_channels=96, kernel_size=3, padding=1),
                ConvBlock(in_channels=96, out_channels=96, kernel_size=3, stride=2),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])


class InceptionLayerC(InceptionLayerBase):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels

        super().__init__()

    def make_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ConvBlock(in_channels=self.in_channels, out_channels=192, kernel_size=1),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=self.bottleneck_channels, kernel_size=1),
                ConvBlock(
                    in_channels=self.bottleneck_channels,
                    out_channels=self.bottleneck_channels,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                ),
                ConvBlock(
                    in_channels=self.bottleneck_channels,
                    out_channels=192,
                    kernel_size=(7, 1),
                    padding=(3, 0),
                ),
            ),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=self.bottleneck_channels, kernel_size=1),
                ConvBlock(
                    in_channels=self.bottleneck_channels,
                    out_channels=self.bottleneck_channels,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                ),
                ConvBlock(
                    in_channels=self.bottleneck_channels,
                    out_channels=self.bottleneck_channels,
                    kernel_size=(7, 1),
                    padding=(3, 0),
                ),
                ConvBlock(
                    in_channels=self.bottleneck_channels,
                    out_channels=self.bottleneck_channels,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                ),
                ConvBlock(
                    in_channels=self.bottleneck_channels,
                    out_channels=192,
                    kernel_size=(7, 1),
                    padding=(3, 0),
                ),
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels=self.in_channels, out_channels=192, kernel_size=1),
            )
        ])


class InceptionLayerD(InceptionLayerBase):
    def __init__(self, in_channels: int):
        self.in_channels = in_channels
        super().__init__()

    def make_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=192, kernel_size=1),
                ConvBlock(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
                ConvBlock(in_channels=192, out_channels=192, kernel_size=(7, 1), padding=(3, 0)),
                ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2),
            ),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=192, kernel_size=1),
                ConvBlock(in_channels=192, out_channels=320, kernel_size=3, stride=2),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])


class ForkedConvLayer(InceptionLayerBase):
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

    def make_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ConvBlock(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, 3), padding=(0, 1)),
            ConvBlock(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3, 1), padding=(1, 0)),
        ])


class InceptionLayerE(InceptionLayerBase):
    def __init__(self, in_channels: int):
        self.in_channels = in_channels

        super().__init__()

    def make_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ConvBlock(in_channels=self.in_channels, out_channels=320, kernel_size=1),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels=self.in_channels, out_channels=192, kernel_size=1),
            ),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=384, kernel_size=1),
                ForkedConvLayer(in_channels=384, out_channels=384),
            ),
            nn.Sequential(
                ConvBlock(in_channels=self.in_channels, out_channels=448, kernel_size=1),
                ConvBlock(in_channels=448, out_channels=384, kernel_size=3, padding=1),
                ForkedConvLayer(in_channels=384, out_channels=384),
            )
        ])


class InceptionV3(nn.Module):
    def __init__(self, dropout: float = 0.5, num_classes: int = 1000):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # Note(pauldb): In https://pytorch.org/assets/images/inception_v3.png, this claims to use kernel_size 3.
            ConvBlock(in_channels=64, out_channels=80, kernel_size=1),
            # Note(pauldb): According to https://pytorch.org/assets/images/inception_v3.png, the next two layers should
            # be instead replaced by ConvBlock(in_channels=80, out_channels=192, kernel_size=3, stride=2).
            ConvBlock(in_channels=80, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Note(pauldb): padding=1 is not listed in the table, but the image size doesn't change.
            # According to https://pytorch.org/assets/images/inception_v3.png, there should be one more convolution
            # block here like: ConvBlock(in_channels=192, out_channels=288, kernel_size=3).
            InceptionLayerA(in_channels=192, num_pool_branch_out_channels=32),
            InceptionLayerA(in_channels=256, num_pool_branch_out_channels=64),
            InceptionLayerA(in_channels=288, num_pool_branch_out_channels=64),
            InceptionLayerB(in_channels=288),
            InceptionLayerC(in_channels=768, bottleneck_channels=128),
            InceptionLayerC(in_channels=768, bottleneck_channels=160),
            InceptionLayerC(in_channels=768, bottleneck_channels=160),
            InceptionLayerC(in_channels=768, bottleneck_channels=192),
            InceptionLayerD(in_channels=768),
            InceptionLayerE(in_channels=1280),
            InceptionLayerE(in_channels=2048),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        batch = batch.to(torch.device("cuda"))
        logits = self.layers(batch.images)
        return F.cross_entropy(logits, batch.classes)

    def eval_forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.layers(images), dim=1)
