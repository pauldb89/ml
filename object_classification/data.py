from __future__ import annotations

import os
import time
from dataclasses import dataclass
from time import perf_counter
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import transforms

from consts import IMAGENET_MEAN
from consts import IMAGENET_STD
from object_classification.distributed_utils import get_rank
from object_classification.distributed_utils import print_once
from object_classification.distributed_utils import world_size


@dataclass
class DatasetConfig:
    name: str
    num_classes: int


IMAGENET_CONFIG = DatasetConfig(name="imagenet", num_classes=1000)
IMAGENETTE_CONFIG = DatasetConfig(name="imagenette", num_classes=10)
DATASETS = {
    "imagenet": IMAGENET_CONFIG,
    "imagenette": IMAGENETTE_CONFIG,
}


class Batch(NamedTuple):
    images: torch.Tensor
    classes: torch.Tensor

    def size(self) -> int:
        return self.classes.numel()

    def to(self, device: torch.device) -> Batch:
        return Batch(
            images=self.images.to(device),
            classes=self.classes.to(device),
        )


def collate_fn(raw_batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    images, classes = zip(*raw_batch)
    return Batch(
        images=torch.stack(images, dim=0),
        classes=torch.tensor(classes),
    )


class DatasetTimer:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print_once(f"Loading dataset took {time.perf_counter() - self.time:.5f} seconds")


def get_train_data_loader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    max_steps: int,
    resize_dim: int
) -> DataLoader:
    with DatasetTimer():
        dataset = ImageFolder(
            root=os.path.join("/datasets", dataset_name, "train"),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(resize_dim),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=RandomSampler(data_source=dataset, num_samples=batch_size * max_steps),
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )


def get_eval_data_loader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    resize_dim: int,
    crop_dim: int
) -> DataLoader:
    with DatasetTimer():
        # We evaluate the models in distributed fashion, so we want to keep approximately 1 / world_size() disjoint
        # examples per worker. Additionally, ImageFolder requires at least one example from each class, so we can't
        # pick the examples non-deterministically (e.g. based on a hash of the filename) since some classes will not
        # be covered in each process.
        def is_valid_file(filename: str) -> bool:
            is_valid_file.calls += 1
            return is_image_file(filename) and is_valid_file.calls % world_size() == get_rank()
        is_valid_file.calls = 0

        dataset = ImageFolder(
            root=os.path.join("/datasets", dataset_name, "val"),
            transform=transforms.Compose([
                transforms.Resize(resize_dim),
                transforms.CenterCrop(crop_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            is_valid_file=is_valid_file,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
