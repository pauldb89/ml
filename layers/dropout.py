import unittest
from argparse import ArgumentParser

import torch
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()

        assert 0 <= dropout_rate <= 1, f"Dropout rate {dropout_rate} must be between [0, 1]"
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        mask = torch.rand_like(x) > self.dropout_rate
        return (x * mask) / (1 - self.dropout_rate)


class MyDropoutTestCase(unittest.TestCase):
    def test_basic(self):
        x = torch.rand(1000, 2048 * 32)
        layer = MyDropout(dropout_rate=0.3)
        layer.train()

        output = layer(x)
        torch.testing.assert_allclose(torch.mean(x) - torch.mean(output), 0, atol=1e-4, rtol=0)
        torch.testing.assert_allclose(torch.count_nonzero(output) / torch.count_nonzero(x), 0.7, atol=1e-4, rtol=0)

        layer.eval()
        torch.testing.assert_allclose(layer(x), x)


if __name__ == "__main__":
    unittest.main()
