from __future__ import annotations

import os
from typing import Callable
from typing import Optional

import PIL
from torch.utils import data
from torch.utils.data.dataset import T_co


class ImageDataset(data.Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()

        self.root = root
        self.filenames = list(sorted(os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index) -> T_co:
        image = PIL.Image.open(os.path.join(self.root, self.filenames[index])).convert("RGB")
        return self.transform(image)

    def __len__(self):
        return len(self.filenames)
