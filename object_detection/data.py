from __future__ import annotations

import math
from random import randint
from random import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

from common.distributed import get_rank
from common.distributed import world_size
from consts import IMAGENET_MEAN
from consts import IMAGENET_STD

Image = Union[np.ndarray, torch.Tensor]
Annotations = List[Dict[str, Any]]


class DetectionTransform:
    def __call__(self, image: Image, labels: Annotations) -> Tuple[Image, Annotations]:
        raise NotImplementedError()


class Compose(DetectionTransform):
    def __init__(self, transform_fns: List[DetectionTransform]):
        super().__init__()
        self.transforms = transform_fns

    def __call__(self, image: Image, labels: Annotations) -> Tuple[Image, Annotations]:
        for transform in self.transforms:
            image, labels = transform(image, labels)
        return image, labels


class FormatInput(DetectionTransform):
    def __init__(self):
        super(FormatInput, self).__init__()
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, image: Image, labels: Annotations) -> Tuple[Image, Annotations]:
        return self.image_transforms(image), labels


class RandomHorizontalFlip(DetectionTransform):
    def __init__(self, p: float = 0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def __call__(self, image: Image, labels: Annotations) -> Tuple[Image, Annotations]:
        if random() > self.p:
            return image, labels

        image_width = image.shape[-1]
        for annotation in labels:
            x, y, w, h = annotation["bbox"]
            annotation["bbox"] = [image_width - x, y, w, h]

        return F.hflip(image), labels
    
    
class RandomResizeShortestEdge(DetectionTransform):
    def __init__(self, min_sizes: List[int], max_size: int):
        super(RandomResizeShortestEdge, self).__init__()
        self.min_sizes = min_sizes
        self.max_size = max_size

    def __call__(self, image: Image, labels: Annotations) -> Tuple[Image, Annotations]:
        min_size = self.min_sizes[randint(0, len(self.min_sizes) - 1)]
        image_height, image_width = image.shape[-2], image.shape[-1]
        r = min(min_size / min(image_height, image_width), self.max_size / max(image_height, image_width))

        new_image_height = int(image_height * r)
        new_image_width = int(image_width * r)

        image = F.resize(image, [new_image_height, new_image_width])
        for annotation in labels:
            x, y, w, h = annotation["bbox"]
            annotation["bbox"] = [
                x * new_image_width / image_width,
                y * new_image_height / image_height,
                w * new_image_width / image_width,
                h * new_image_height / image_height,
            ]

        return image, labels


class CocoDetectionDataset(CocoDetection):
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform_fns: Optional[Callable],
        keep_annotated_images_only: bool,
    ):
        super().__init__(root=root_dir, annFile=annotation_file, transforms=transform_fns)

        if keep_annotated_images_only:
            self.ids = [image_id for image_id in sorted(self.coco.imgs.keys()) if len(self.coco.imgToAnns[image_id])]
        else:
            self.ids = list(sorted(self.coco.imgs.keys()))


class Batch(NamedTuple):
    images: torch.Tensor
    labels: List[List[Dict[str, Any]]]

    def to(self, device: torch.device) -> Batch:
        return Batch(images=self.images.to(device), labels=self.labels)

    def size(self) -> int:
        return len(self.labels)


# TODO(pauldb): Test
def pad_images(images: List[torch.Tensor], size_multiplier: int = 64) -> torch.Tensor:
    max_height, max_width = 0, 0
    for image in images:
        max_height = max(max_height, image.shape[-2])
        max_width = max(max_width, image.shape[-1])

    max_height = int(math.ceil(max_height / size_multiplier) * size_multiplier)
    max_width = int(math.ceil(max_width / size_multiplier) * size_multiplier)

    batched_images = torch.zeros(len(images), 3, max_height, max_width, dtype=torch.half)

    for i, image in enumerate(images):
        batched_images[i, :, :image.shape[-2], :image.shape[-1]] = image

    return batched_images.contiguous()


def collate_fn(data: List[Tuple[torch.Tensor, List[Dict[str, Any]]]]) -> Batch:
    images, labels = zip(*data)
    return Batch(images=pad_images(images), labels=list(labels))


def create_data_loader(root_dir: str, annotation_file: str, batch_size: int, is_train: bool) -> DataLoader:
    train_dataset = CocoDetectionDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform_fns=Compose(transform_fns=[
            FormatInput(),
            RandomHorizontalFlip(),
            RandomResizeShortestEdge(min_sizes=[640, 672, 704, 736, 768, 800], max_size=1333)
            # RandomResizeShortestEdge(min_sizes=[704], max_size=1333)
        ]),
        keep_annotated_images_only=is_train,
    )

    generator = torch.Generator()
    generator.manual_seed(0)

    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=train_dataset, num_replicas=world_size(), rank=get_rank()),
        collate_fn=collate_fn,
        drop_last=is_train,
        generator=generator,
    )
