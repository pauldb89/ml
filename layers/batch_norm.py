import unittest
from typing import Optional

import torch
from torch import nn


class MyBatchNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum

        self.weights = nn.Parameter(data=torch.ones(num_channels, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(num_channels, dtype=torch.float), requires_grad=True)

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 4
        assert x.size()[1] == self.num_channels

        if self.training:
            batch_mean = torch.mean(x, dim=(0, 2, 3))
            batch_var = torch.var(x, dim=(0, 2, 3), unbiased=False)
            num_el = torch.numel(x) / self.num_channels
            batch_var_unbiased = batch_var * num_el / (num_el - 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_unbiased
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        batch_std = torch.sqrt(batch_var) + self.eps

        norm_x = (x - batch_mean.view(self.num_channels, 1, 1)) / (batch_std.view(self.num_channels, 1, 1))
        return self.weights.view(self.num_channels, 1, 1) * norm_x + self.bias.view(self.num_channels, 1, 1)


class MyBatchNorm2dTestCase(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)
        num_channels = 10
        ref_layer = nn.BatchNorm2d(num_features=num_channels)
        layer = MyBatchNorm2d(num_channels=num_channels)

        for _ in range(10):
            x = torch.rand(3, num_channels, 5, 6)
            ref_output = ref_layer(x)

            torch.testing.assert_allclose(torch.mean(ref_output, dim=(0, 2, 3)), torch.zeros(num_channels))
            torch.testing.assert_allclose(torch.std(ref_output, dim=(0, 2, 3), unbiased=False), torch.ones(num_channels))

            output = layer(x)
            torch.testing.assert_allclose(torch.mean(output, dim=(0, 2, 3)), torch.zeros(num_channels))
            torch.testing.assert_allclose(torch.std(output, dim=(0, 2, 3), unbiased=False), torch.ones(num_channels))

            torch.testing.assert_allclose(output, ref_output)
            torch.testing.assert_allclose(ref_layer.running_mean, layer.running_mean)
            torch.testing.assert_allclose(ref_layer.running_var, layer.running_var)

        ref_layer.eval()
        layer.eval()

        for _ in range(10):
            x = torch.rand(3, num_channels, 5, 6)
            ref_output = ref_layer(x)
            output = layer(x)
            torch.testing.assert_allclose(output, ref_output)


if __name__ == "__main__":
    unittest.main()
