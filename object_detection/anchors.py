import math
from typing import Tuple, List

import torch
from torch import nn


class AnchorGenerator(nn.Module):
    def __init__(self, resolutions: Tuple[int, ...], strides: Tuple[int, ...], aspect_ratios: Tuple[float, ...]):
        super().__init__()

        assert len(resolutions) == len(strides), f"{len(resolutions)} resolutions vs {len(strides)} strides"
        self.resolutions = resolutions
        self.strides = strides
        self.aspect_ratios = aspect_ratios

        anchor_dims = []
        for resolution in resolutions:
            level_dims = []
            for aspect_ratio in aspect_ratios:
                width = resolution / math.sqrt(aspect_ratio)
                height = resolution * math.sqrt(aspect_ratio)
                level_dims.append([width, height])
            anchor_dims.append(level_dims)

        self.register_buffer("anchor_dims", torch.tensor(anchor_dims))

    def forward(self, max_width: int, max_height: int) -> List[torch.Tensor]:
        """
        :param max_width: Width of images in a batch (all are padded to have the same size).
        :param max_height: Height of images in a batch (all are padded to have the same size).
        :return: A list of L tensors of size Ri x 4 representing the anchors.
        """
        anchors = []
        for level, stride in enumerate(self.strides):
            x = torch.arange(0, max_width, stride, device="cuda")
            y = torch.arange(0, max_height, stride, device="cuda")
            y, x = torch.meshgrid(y, x, indexing="ij")

            offsets = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
            anchor_dims = self.anchor_dims[level].view(1, -1, 2)
            anchors.append(
                torch.cat(
                    [
                        (offsets.view(-1, 1, 2) - anchor_dims / 2).view(-1, 2),
                        anchor_dims.repeat(1, offsets.shape[0], 1).view(-1, 2),
                    ],
                    dim=1,
                )
            )

        return anchors


def sample_anchors(anchor_mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Pick k random unique anchor IDs matching the anchor mask.
    :param anchor_mask: A boolean tensor of size R representing the mask.
    :param k: An integer representing the number of anchor IDs to pick.
    :return: A tensor of size k containing the sampled anchor IDs.
    """
    anchor_ids = torch.nonzero(anchor_mask).view(-1)
    return anchor_ids[torch.randperm(anchor_ids.numel())[:k]]


def sample_labels(
    mask: torch.Tensor,
    num_samples: int,
    positive_fraction: float,
    negative_class: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    positive_mask = torch.logical_and(mask >= 0, mask != negative_class)
    negative_mask = mask == negative_class

    num_positives = int(min(positive_fraction * num_samples, torch.sum(positive_mask).item()))
    num_negatives = int(min(num_samples - num_positives, torch.sum(negative_mask).item()))

    return sample_anchors(positive_mask, num_positives), sample_anchors(negative_mask, num_negatives)
