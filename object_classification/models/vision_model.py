from __future__ import annotations

from typing import Protocol

import torch

from object_classification.data import Batch


class VisionModel(Protocol):
    def __call__(self, batch: Batch) -> torch.Tensor:
        ...

    def forward(self, batch: Batch) -> torch.Tensor:
        ...

    def eval_forward(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def train(self) -> None:
        ...

    def eval(self) -> None:
        ...
