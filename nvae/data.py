import time
from typing import List
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.datasets import CelebA
from torchvision.transforms import transforms

from common.distributed import print_once


def drop_labels(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([images for images, _ in batch], dim=0)


class CropCelebA64:
    def __call__(self, img: Image):
        assert img.size == (178, 218)
        return img.crop((15, 40, 178 - 15, 218 - 30))


def create_train_data_loader(root: str, batch_size: int, num_workers: int = 5) -> DataLoader:
    start_time = time.time()
    dataset = CelebA(
        root=root,
        split="train",
        transform=transforms.Compose([
            CropCelebA64(),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    )
    print_once(f"Loading dataset took {time.time() - start_time} seconds")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=dataset, shuffle=True),
        num_workers=num_workers,
        drop_last=True,
        collate_fn=drop_labels,
    )


def create_eval_data_loader(root: str, batch_size: int, num_workers: int = 5) -> DataLoader:
    start_time = time.time()
    dataset = CelebA(
        root=root,
        split="valid",
        transform=transforms.Compose([
            CropCelebA64(),
            transforms.Resize(64),
            transforms.ToTensor(),
        ]),
    )
    print_once(f"Loading dataset took {time.time() - start_time} seconds")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=dataset, shuffle=False),
        num_workers=num_workers,
        drop_last=False,
        collate_fn=drop_labels,
    )
