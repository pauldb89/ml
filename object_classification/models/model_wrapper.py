import torch
from torch import nn


class PretrainedWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def eval_forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class ModelAverageWrapper(torch.optim.swa_utils.AveragedModel):
    def eval_forward(self, *args, **kwargs):
        return self.module.eval_forward(*args, **kwargs)
