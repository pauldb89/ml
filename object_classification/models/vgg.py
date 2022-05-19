from dataclasses import dataclass
from typing import Sequence

from torch import nn

from object_classification.models.vision_model import VisionModel


@dataclass
class VGGLayerGroupConfig:
    num_conv_layers: int
    out_channels: int


VGG_16_CONFIG = (
    VGGLayerGroupConfig(num_conv_layers=2, out_channels=64),
    VGGLayerGroupConfig(num_conv_layers=2, out_channels=128),
    VGGLayerGroupConfig(num_conv_layers=3, out_channels=256),
    VGGLayerGroupConfig(num_conv_layers=3, out_channels=512),
    VGGLayerGroupConfig(num_conv_layers=3, out_channels=512),
)


class VGG(VisionModel):
    def __init__(
        self,
        layer_group_configs: Sequence[VGGLayerGroupConfig] = VGG_16_CONFIG,
        batch_norm: bool = True,
        dropout: float = 0.5,
        num_classes: int = 1000,
    ):
        super().__init__(layer_group_configs, batch_norm, dropout, num_classes)

    def make_layers(self, layer_group_configs, batch_norm, dropout, num_classes) -> nn.Module:
        in_channels = 3
        layer_groups = []
        for layer_group_config in layer_group_configs:
            layer_group = []
            for _ in range(layer_group_config.num_conv_layers):
                layer_group.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=layer_group_config.out_channels,
                        kernel_size=(3, 3),
                        padding=1,
                    ),
                )
                if batch_norm:
                    layer_group.append(nn.BatchNorm2d(layer_group_config.out_channels))
                layer_group.append(nn.ReLU())
                in_channels = layer_group_config.out_channels

            layer_group.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            layer_groups.append(nn.Sequential(*layer_group))

        layers = nn.Sequential(
            *layer_groups,
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

        for layer in layers.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

        return layers
