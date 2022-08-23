from __future__ import annotations

from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data.dataset import T_co
from torchvision.transforms import transforms

from common.consts import IMAGENET_MEAN
from common.consts import IMAGENET_STD
from common.data import ImageDataset


class ImagePairDataset(data.Dataset):
    def __init__(self, content_dataset: ImageDataset, style_dataset: ImageDataset):
        super().__init__()
        self.content_dataset = content_dataset
        self.style_dataset = style_dataset

    def __getitem__(self, index) -> T_co:
        return self.content_dataset[index], self.style_dataset[index]

    def __len__(self):
        return min(len(self.content_dataset), len(self.style_dataset))


class Batch(NamedTuple):
    content_images: torch.Tensor
    style_images: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(content_images=self.content_images.to(device), style_images=self.style_images.to(device))

    def __len__(self) -> int:
        return self.content_images.shape[0]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    content_images, style_images = zip(*batch)
    return Batch(content_images=torch.stack(content_images, dim=0), style_images=torch.stack(style_images, dim=0))


def create_train_data_loader(content_root_dir: str, style_root_dir: str, batch_size: int, num_workers: int = 5):
    transform = transforms.Compose([
        transforms.Resize(size=512),
        transforms.RandomResizedCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    content_dataset = ImageDataset(root=content_root_dir, transform=transform)
    style_dataset = ImageDataset(root=style_root_dir, transform=transform)
    dataset = ImagePairDataset(content_dataset, style_dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=dataset),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


def create_eval_data_loader(content_root_dir: str, style_root_dir: str, batch_size: int, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    content_dataset = ImageDataset(root=content_root_dir, transform=transform)
    style_dataset = ImageDataset(root=style_root_dir, transform=transform)
    dataset = ImagePairDataset(content_dataset, style_dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=dataset, shuffle=False),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
