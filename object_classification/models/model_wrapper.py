import torch
from torch import nn


class PretrainedWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def eval_forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
