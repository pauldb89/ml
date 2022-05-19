from __future__ import annotations

from typing import Any
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from object_classification.data import Batch


class VisionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.layers = self.make_layers(*args, **kwargs)

    def make_layers(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError("make_layers not implemented")

    def forward(self, batch: Batch) -> Dict[str, Any]:
        batch = batch.to(torch.device("cuda"))
        logits = self.layers(batch.images)
        return {
            "loss": F.cross_entropy(logits, batch.classes),
        }

    def eval_forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.layers(images), dim=1)
