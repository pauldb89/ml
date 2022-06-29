import math
from typing import Tuple

import numpy as np
import torch


class AnchorGenerator:
    def __init__(self, resolutions: Tuple[int, ...], strides: Tuple[int, ...], aspect_ratios: Tuple[float, ...]):
        assert len(resolutions) == len(strides), f"{len(resolutions)} resolutions vs {len(strides)} strides"
        self.resolutions = resolutions
        self.strides = strides
        self.aspect_ratios = aspect_ratios

    def generate(self, max_height: int, max_width: int) -> torch.Tensor:
        """
        :param max_height: Height of images in a batch (all are padded to have the same size).
        :param max_width: Width of images in a batch (all are padded to have the same size).
        :return: A tensor of size R x 4 representing the anchors.
        """
        anchors = []
        for resolution, stride in zip(self.resolutions, self.strides):
            for x in range(0, max_width, stride):
                for y in range(0, max_height, stride):
                    for aspect_ratio in self.aspect_ratios:
                        width = resolution * math.sqrt(aspect_ratio)
                        height = resolution / math.sqrt(aspect_ratio)
                        x1 = max(0.0, x - width / 2)
                        y1 = max(0.0, y - height / 2)
                        x2 = min(x + width / 2, max_width)
                        y2 = min(y + height / 2, max_height)
                        anchors.append([x1, y1, x2 - x1, y2 - y1])

        return torch.tensor(anchors)


def sample_anchors(anchor_mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Pick k random unique anchor IDs matching the anchor mask.
    :param anchor_mask: A boolean tensor of size R representing the mask.
    :param k: An integer representing the number of anchor IDs to pick.
    :return: A tensor of size k containing the sampled anchor IDs.
    """
    anchor_ids = np.flatnonzero(anchor_mask.numpy())
    return torch.from_numpy(np.random.choice(anchor_ids, size=min(anchor_ids.size, k), replace=False))
