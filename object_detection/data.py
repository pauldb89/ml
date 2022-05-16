import math
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms

from consts import IMAGENET_MEAN
from consts import IMAGENET_STD
from common.distributed import get_rank
from common.distributed import world_size


# TODO(pauldb): Test
def pad_images(images: List[torch.Tensor], size_multiplier: int = 32) -> torch.Tensor:
    max_height, max_width = 0, 0
    for image in images:
        max_height = max(max_height, image.shape[-2])
        max_width = max(max_width, image.shape[-1])

    max_height = int(math.ceil(max_height / size_multiplier) * size_multiplier)
    max_width = int(math.ceil(max_width / size_multiplier) * size_multiplier)

    batched_images = torch.zeros(len(images), 3, max_height, max_width)

    for i, image in enumerate(images):
        batched_images[i, :, :image.shape[-2], :image.shape[-1]] = image

    return batched_images.contiguous()


def create_data_loader(root_dir: str, annotation_file: str, batch_size: int, is_train: bool) -> DataLoader:
    train_dataset = CocoDetection(
        root=root_dir,
        annFile=annotation_file,
        transform=transforms.Compose([
            transforms.Resize(600),
            transforms.RandomHorizontalFlip(p=0.5 if is_train else 0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    )
    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=train_dataset, num_replicas=world_size(), rank=get_rank()),
        collate_fn=lambda images, labels: (pad_images(images), labels),
        drop_last=is_train,
    )
