import torch
from torch import nn
from torchvision.ops import SqueezeExcitation


class EncoderCell(nn.Module):
    """
    An encoder cell as depicted in section 3.1.2 of https://arxiv.org/pdf/2007.03898.pdf.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.cell = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.se = SqueezeExcitation(input_channels=out_channels, squeeze_channels=out_channels // 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual(x) + self.cell(x)
        return self.se(x)


class DecoderCell(nn.Module):
    """
    A decoder cell as depicted in section 3.1.1 of https://arxiv.org/pdf/2007.03898.pdf.
    """
    def __init__(self, channels: int, extension_factor: int):
        super().__init__()

        # TODO(pauldb): Handle up-sampling.
        extended_channels = channels * extension_factor

        self.cell = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            nn.Conv2d(in_channels=channels, out_channels=extended_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=extended_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=extended_channels,
                out_channels=extended_channels,
                kernel_size=5,
                groups=extended_channels,
            ),
            nn.BatchNorm2d(num_features=extended_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=extended_channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels),
        )

        self.ce = SqueezeExcitation(input_channels=channels, squeeze_channels=channels // 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(x + self.cell(x))


class NVAE(nn.Module):
    def __init__(self):
        super().__init__()


