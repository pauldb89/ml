from __future__ import annotations

import copy
import math
from random import randint
from random import random
from typing import Any, Iterator, Iterable, Sized
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import PIL.Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms, ToTensor, Normalize
import torchvision.transforms.functional as F

from common.distributed import get_rank
from common.distributed import world_size

Image = Union[np.ndarray, torch.Tensor, PIL.Image.Image]
Label = Dict[str, Any]
Labels = List[Label]


class DetectionTransform:
    def __call__(self, image: Image, labels: Labels) -> Tuple[Image, Labels]:
        raise NotImplementedError()


class Compose(DetectionTransform):
    def __init__(self, transform_fns: List[DetectionTransform]):
        super().__init__()
        self.transforms = transform_fns

    def __call__(self, image: Image, label: Label) -> Tuple[Image, Label]:
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class ReadImage(DetectionTransform):
    def __call__(self, image: Image, label: Label) -> Tuple[Image, Label]:
        image = np.array(image, np.uint8, copy=True)
        image = np.ascontiguousarray(image[:, :, ::-1])
        return image, label


class Transform(DetectionTransform):
    def __init__(self):
        super().__init__()
        self.image_transforms = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image, labels: Labels) -> Tuple[Image, Labels]:
        return self.image_transforms(image), labels


class RandomHorizontalFlip(DetectionTransform):
    def __init__(self, p: float = 0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def __call__(self, image: Image, label: Label) -> Tuple[Image, Label]:
        if random() > self.p:
            return image, label

        image_width = image.shape[-1]
        for annotation in label["annotations"]:
            x, y, w, h = annotation["bbox"]
            annotation["bbox"] = [image_width - x - w, y, w, h]

        return F.hflip(image), label
    
    
class RandomResizeShortestEdge(DetectionTransform):
    def __init__(self, min_sizes: List[int], max_size: int):
        super(RandomResizeShortestEdge, self).__init__()
        self.min_sizes = min_sizes
        self.max_size = max_size

    def __call__(self, image: Image, label: Label) -> Tuple[Image, Label]:
        min_size = self.min_sizes[randint(0, len(self.min_sizes) - 1)]
        image_height, image_width = image.shape[0], image.shape[1]
        r = min(min_size / min(image_height, image_width), self.max_size / max(image_height, image_width))

        new_image_height = round(image_height * r)
        new_image_width = round(image_width * r)

        pil_image = PIL.Image.fromarray(image)

        pil_image = pil_image.resize((new_image_width, new_image_height), PIL.Image.BILINEAR)
        image = np.asarray(pil_image)

        for annotation in label["annotations"]:
            x, y, w, h = annotation["bbox"]
            annotation["bbox"] = [
                x * new_image_width / image_width,
                y * new_image_height / image_height,
                w * new_image_width / image_width,
                h * new_image_height / image_height,
            ]

        return image, label


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

        self.categories = list(sorted(self.coco.getCatIds()))
        self.category_mapping = {cat_id: idx for idx, cat_id in enumerate(self.categories)}

    def _load_target(self, x: int) -> Label:
        image_metadata = self.coco.imgs[x]
        original_annotations = super()._load_target(x)

        filtered_annotations = []
        for annotation in original_annotations:
            if annotation["iscrowd"] == 0:
                annotation_copy = copy.deepcopy(annotation)
                annotation_copy["category_id"] = self.category_mapping[annotation_copy["category_id"]]
                filtered_annotations.append(annotation_copy)

        return {
            "annotations": filtered_annotations,
            "original_height": image_metadata["height"],
            "original_width": image_metadata["width"],
            "image_id": x,
        }


class Batch(NamedTuple):
    images: torch.Tensor
    image_sizes: List[Tuple[int, int]]
    labels: Optional[Labels]

    def to(self, device: torch.device) -> Batch:
        return Batch(images=self.images.to(device), labels=self.labels, image_sizes=self.image_sizes)

    def size(self) -> int:
        return len(self.image_sizes)


# TODO(pauldb): Test
def pad_images(images: List[torch.Tensor], size_multiplier: int = 32) -> torch.Tensor:
    max_height, max_width = 0, 0
    for image in images:
        max_height = max(max_height, image.shape[-2])
        max_width = max(max_width, image.shape[-1])

    max_height = int(math.ceil(max_height / size_multiplier) * size_multiplier)
    max_width = int(math.ceil(max_width / size_multiplier) * size_multiplier)

    batched_images = torch.zeros(len(images), 3, max_height, max_width, dtype=torch.float32)

    for i, image in enumerate(images):
        batched_images[i, :, :image.shape[-2], :image.shape[-1]] = image

    return batched_images.contiguous()


def collate_fn(data: List[Tuple[torch.Tensor, Label]]) -> Batch:
    images, labels = zip(*data)
    image_sizes = [(image.shape[-1], image.shape[-2]) for image in images]
    return Batch(images=pad_images(images), labels=list(labels), image_sizes=image_sizes)


class AspectRatioDataLoader(Iterable[Batch], Sized):
    def __init__(self, example_loader: DataLoader, batch_size: int):
        super().__init__()
        self.example_loader = example_loader
        self.batch_size = batch_size
        self.buffers = [[] for _ in range(2)]

    def __iter__(self) -> Iterator[Batch]:
        for examples in self.example_loader:
            for image, label in examples:
                group_id = 0 if image.shape[-2] < image.shape[-1] else 1
                self.buffers[group_id].append((image, label))
                if len(self.buffers[group_id]) == self.batch_size:
                    yield collate_fn(self.buffers[group_id])
                    self.buffers[group_id] = []

    def __len__(self) -> int:
        return len(self.example_loader)


def create_train_data_loader(
    root_dir: str,
    annotation_file: str,
    batch_size: int,
    max_steps: int,
    num_workers: int = 4,
) -> AspectRatioDataLoader:
    dataset = CocoDetectionDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform_fns=Compose(transform_fns=[
            ReadImage(),
            RandomResizeShortestEdge(min_sizes=[640, 672, 704, 736, 768, 800], max_size=1333),
            Transform(),
            RandomHorizontalFlip(),
        ]),
        keep_annotated_images_only=True,
    )

    return AspectRatioDataLoader(
        DataLoader(
            dataset=dataset,
            sampler=RandomSampler(data_source=dataset, replacement=False, num_samples=(max_steps + 2) * batch_size),
            num_workers=num_workers,
            collate_fn=lambda x: x,
        ),
        batch_size=batch_size,
    )


def create_eval_data_loader(
    root_dir: str,
    annotation_file: str,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CocoDetectionDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform_fns=Compose(transform_fns=[
            ReadImage(),
            RandomResizeShortestEdge(min_sizes=[800], max_size=1333),
            Transform(),
        ]),
        keep_annotated_images_only=False,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices=range(get_rank(), len(dataset), world_size())),
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )
