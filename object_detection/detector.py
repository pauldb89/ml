import functools
import itertools
import math
import time
from collections import Counter
from collections import defaultdict
from collections import deque
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
import torchvision.models
import torchvision.ops as O
from torch import nn
from torch.nn import functional as F

from object_detection.anchors import AnchorGenerator
from object_detection.anchors import sample_anchors
from object_detection.bounding_box import BBox
from object_detection.data import Batch


"""
Notations:
B - number of images per batch.
L - Number of resolutions (number of levels in the feature pyramid).
H - Height of all images in a batch.
Hi - Height of the feature map at level i in the feature pyramid (0 -> P_2, 1 -> P_3, ...)
W - Width of all images in a batch.
Wi - Width of the feature map at level i in the feature pyramid (0 -> P_2, 1 -> P_3, ...)
C - Number of channels for feature maps.
A - Number of aspect ratios.
K - Number of classes.
R - Total number of anchors / region proposals for an image = A * sum_{i=1}^L H_i * W_i
S - Number of region proposals sampled for computing the RPN loss.
Pi - Number of ROIs after NMS for each image i in the batch.
TODO(pauldb): Explain how the anchor ID mapping works.
"""


def timed(fn: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        ret = fn(self, *args, **kwargs)
        prefix = "train" if self.training else "eval"
        self.time_stats[f"{prefix}_{fn.__name__}"].append(time.time() - start_time)
        return ret
    return wrapper


MAX_LEVEL = 5
STRIDE_LEVEL_FACTOR = 4
ANCHOR_SIZE_LEVEL_FACTOR = 32


def compute_ious(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(a[:, 0].view(-1, 1), b[:, 0])
    iy1 = torch.maximum(a[:, 1].view(-1, 1), b[:, 1])
    ix2 = torch.minimum((a[:, 0] + a[:, 2]).view(-1, 1), b[:, 0] + b[:, 2])
    iy2 = torch.minimum((a[:, 1] + a[:, 3]).view(-1, 1), b[:, 1] + b[:, 3])
    i_area = torch.maximum(ix2 - ix1, torch.tensor(0.0)) * torch.maximum(iy2 - iy1, torch.tensor(0.0))
    a_area = a[:, 2] * a[:, 3]
    b_area = b[:, 2] * b[:, 3]
    iou = i_area / (a_area.view(-1, 1) + b_area - i_area)
    return iou


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        relu: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        ]

        if relu:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RPNLabels(NamedTuple):
    # A tensor of shape B x S representing whether the sampled ROI corresponds to an object or not.
    objectness_labels: torch.Tensor
    # A tensor of shape B x S representing the anchor IDs matching the sampled ROIs.
    anchor_ids: torch.Tensor
    # A tensor of shape B x S x 4 representing non-centered target bboxes for the anchor IDs with objectness label of 1.
    target_bboxes: torch.Tensor


def center_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    x = bboxes[..., 0] + bboxes[..., 2] / 2
    y = bboxes[..., 1] + bboxes[..., 3] / 2
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    return torch.stack([x, y, w, h], dim=-1)


def uncenter_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    x = bboxes[..., 0] - bboxes[..., 2] / 2
    y = bboxes[..., 1] - bboxes[..., 3] / 2
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    return torch.stack([x, y, w, h], dim=-1)


def scale_bboxes(bboxes: torch.Tensor, centered_anchors: torch.Tensor) -> torch.Tensor:
    assert bboxes.shape == centered_anchors.shape, (bboxes.shape, centered_anchors.shape)

    x = bboxes[..., 0] * centered_anchors[..., 2] + centered_anchors[..., 0]
    y = bboxes[..., 1] * centered_anchors[..., 3] + centered_anchors[..., 1]
    w = centered_anchors[..., 2] * torch.exp(bboxes[..., 2])
    h = centered_anchors[..., 3] * torch.exp(bboxes[..., 3])
    return torch.stack([x, y, w, h], dim=-1)


def parse_annotated_bboxes(annotations: List[Dict[str, Any]]) -> torch.Tensor:
    bboxes = []
    for annotation in annotations:
        bbox = annotation["bbox"]
        bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
    return torch.tensor(bboxes)


def parse_annotated_categories(annotations: List[Dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([annotation["category_id"] for annotation in annotations])


class Detector(nn.Module):
    def __init__(
        self,
        resolutions: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (4, 8, 16, 32, 64),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        rpn_positive_iou_threshold: float = 0.7,
        rpn_negative_iou_threshold: float = 0.3,
        num_images_per_batch: int = 2,
        rpn_regions_per_batch: int = 256,
        rpn_positive_regions_per_batch: int = 128,
        nms_iou_threshold: float = 0.7,
        max_train_nms_candidates: int = 2000,
        max_eval_nms_candidates: int = 1000,
        max_train_rois_per_image: int = 1000,
        max_eval_rois_per_image: int = 1000,
        bbox_loss_weight: float = 10.0,
        detection_iou_threshold: float = 0.5,
        num_classes: int = 80,
    ):
        super().__init__()

        self.resolutions = resolutions
        self.aspect_ratios = aspect_ratios
        self.anchor_generator = AnchorGenerator(resolutions=resolutions, strides=strides, aspect_ratios=aspect_ratios)

        self.rpn_positive_iou_threshold = rpn_positive_iou_threshold
        self.rpn_negative_iou_threshold = rpn_negative_iou_threshold
        self.num_images_per_batch = num_images_per_batch
        self.rpn_regions_per_batch = rpn_regions_per_batch
        self.rpn_positive_regions_per_batch = rpn_positive_regions_per_batch
        self.nms_iou_threshold = nms_iou_threshold
        self.max_train_rois_per_image = max_train_rois_per_image
        self.max_eval_rois_per_image = max_eval_rois_per_image
        self.max_train_nms_candidates = max_train_nms_candidates
        self.max_eval_nms_candidates = max_eval_nms_candidates
        self.bbox_loss_weight = bbox_loss_weight
        self.detection_iou_threshold = detection_iou_threshold
        self.num_classes = num_classes

        self.time_stats = defaultdict(lambda: deque(maxlen=100))

        self.backbone = torchvision.models.resnet50(pretrained=True)

        target_layers = [
            self.backbone.layer1[-1].conv3,
            self.backbone.layer2[-1].conv3,
            self.backbone.layer3[-1].conv3,
            self.backbone.layer4[-1].conv3,
        ]
        self.tracked_feature_maps: List[Optional[torch.Tensor]] = [None] * len(target_layers)
        for i, target_layer in enumerate(target_layers):
            def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor, layer_index: int) -> None:
                self.tracked_feature_maps[layer_index] = output

            target_layer.register_forward_hook(hook=functools.partial(hook_fn, layer_index=i))

        self.lateral_connections = nn.ModuleList([
            ConvBlock(
                in_channels=target_layer.out_channels,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for target_layer in target_layers
        ])
        # Conv transforms which help reduce the aliasing effect of up-sampling. Interesting
        # that they are not added directly on the top-down path.
        self.feature_map_transforms = nn.ModuleList([
            ConvBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                relu=False,
            )
            for _ in range(len(target_layers) + 1)
        ])

        self.region_proposal_head = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(
                in_channels=256,
                out_channels=5 * len(self.aspect_ratios),
                kernel_size=1,
                stride=1,
                padding=0,
                relu=False,
            ),
        )

        self.detection_trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.classification_head = nn.Linear(1024, num_classes + 1)
        self.regression_head = nn.Linear(1024, 4 * num_classes)

    def compute_unscaled_proposals(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param images: A tensor of images of size B x 3 x H x W.
        :return: A tuple:
            - feature_maps: A list of L tensors of shape B x C x Hi x Wi
            - unscaled_proposals: A tensor of shape B x R x (1 + 4).
        """
        self.backbone(images)

        feature_maps = [torch.tensor(0) for _ in range(len(self.tracked_feature_maps))]
        for i, tracked_feature_map in reversed(list(enumerate(self.tracked_feature_maps))):
            x = self.lateral_connections[i](tracked_feature_map)
            if i+1 < len(feature_maps):
                feature_maps[i] = x + F.interpolate(feature_maps[i+1], scale_factor=2, mode="nearest")
            else:
                feature_maps[i] = x

        feature_maps.append(F.interpolate(feature_maps[-1], scale_factor=0.5))

        unscaled_proposals = []
        for i, feature_map in enumerate(feature_maps):
            proposal = self.region_proposal_head(self.feature_map_transforms[i](feature_map))
            # Shape B x (A * (1 + 4)) x H x W ==> B x H x W x (A * (4 + 1)).
            proposal = torch.permute(proposal, dims=[0, 2, 3, 1])
            # Shape B x H x W x (A * (4 + 1)) ==> B x (H * W * A) x (4 + 1)
            unscaled_proposals.append(proposal.reshape(images.shape[0], -1, 5))

        # Shape B x R x 5.
        unscaled_proposals = torch.cat(unscaled_proposals, dim=1)

        return feature_maps, unscaled_proposals

    @timed
    def compute_proposal_labels(self, anchors: torch.Tensor, labels: List[List[Dict[str, Any]]]) -> RPNLabels:
        """
        :param anchors: A tensor of size R x 4.
        :param labels: A list of size B containing lists with the ground truth annotations for each image.
        :return: RPN labels.
        """
        objectness_labels = []
        anchor_ids = []
        target_bboxes = []
        for image_labels in labels:
            gt_bboxes = parse_annotated_bboxes(image_labels)
            ious = compute_ious(anchors, gt_bboxes)

            best_ious, best_gts = ious.max(dim=1)
            print("best ious", torch.topk(best_ious, k=10))
            positive_mask = best_ious >= self.rpn_positive_iou_threshold
            positive_mask[ious.max(dim=0).indices] = 1

            negative_mask = torch.logical_and(
                torch.logical_not(positive_mask),
                best_ious <= self.rpn_negative_iou_threshold,
            )

            print(f"ground truth boxes", gt_bboxes)
            print(f"positive label pool {torch.sum(positive_mask)}, negative label pool {torch.sum(negative_mask)}")

            positive_anchor_ids = sample_anchors(anchor_mask=positive_mask, k=self.rpn_positive_regions_per_batch)
            negative_anchor_ids = sample_anchors(
                anchor_mask=negative_mask,
                k=self.rpn_regions_per_batch - positive_anchor_ids.numel()
            )

            sampled_anchor_ids = torch.cat([positive_anchor_ids, negative_anchor_ids], dim=0)
            objectness_labels.append(
                torch.cat(
                    [
                        torch.ones_like(positive_anchor_ids, dtype=torch.float),
                        torch.zeros_like(negative_anchor_ids, dtype=torch.float),
                    ],
                    dim=0
                )
            )
            anchor_ids.append(sampled_anchor_ids)
            target_bboxes.append(gt_bboxes[best_gts[sampled_anchor_ids]])

        return RPNLabels(
            # B x S
            objectness_labels=torch.stack(objectness_labels, dim=0),
            # B x S
            anchor_ids=torch.stack(anchor_ids, dim=0),
            # B x S x 4
            target_bboxes=torch.stack(target_bboxes, dim=0),
        )

    @timed
    def compute_regression_box_loss(
        self,
        unscaled_predicted_bboxes: torch.Tensor,
        ground_truth_bboxes: torch.Tensor,
        reference_bboxes: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """
        In the following X is either B x S (for the RPN loss) or P (for the detection bounding box loss).
        :param unscaled_predicted_bboxes: X x 4 tensor representing the centered bbox coordinates of the predictions.
        # TODO(pauldb): If these comments are true, why do we center the bboxes in this function?
        :param ground_truth_bboxes: X x 4 tensor representing the centered bbox coordinates of the ground truth labels.
        :param reference_bboxes: X x 4 tensor representing the centered coordinates of the reference bboxes.
        :param weights: X tensors representing individual weights (or mask).
        :return:
        """
        unscaled_predicted_bboxes = unscaled_predicted_bboxes.cuda()
        ground_truth_bboxes = ground_truth_bboxes.cuda()
        reference_bboxes = reference_bboxes.cuda()
        weights = weights.cuda() if isinstance(weights, torch.Tensor) else weights

        ground_truth_bboxes = center_bboxes(ground_truth_bboxes)
        reference_bboxes = center_bboxes(reference_bboxes)

        location_loss = F.smooth_l1_loss(
            unscaled_predicted_bboxes[..., 0:2],
            (ground_truth_bboxes[..., 0:2] - reference_bboxes[..., 0:2]) / reference_bboxes[..., 2:4],
            reduction="none",
        )
        size_loss = F.smooth_l1_loss(
            unscaled_predicted_bboxes[..., 2:4],
            torch.log(ground_truth_bboxes[..., 2:4] / reference_bboxes[..., 2:4]),
            reduction="none",
        )
        bbox_loss = torch.mean(
            torch.cat([location_loss, size_loss], dim=-1).mean(dim=-1) * weights
        )
        return bbox_loss

    @timed
    def compute_rpn_loss(
        self,
        unscaled_proposals: torch.Tensor,
        anchors: torch.Tensor,
        rpn_labels: RPNLabels,
    ) -> torch.Tensor:
        """
        :param unscaled_proposals: A tensor of region proposals of size B x R x (1 + 4).
        :param anchors: A tensor of anchors of size R x 4.
        :param rpn_labels: A RPNLabels struct containing a sampled set of labels for computing the loss.
        :return: A scalar tensor representing the loss.
        """
        anchor_index = torch.unsqueeze(rpn_labels.anchor_ids, dim=2).repeat(1, 1, 5).cuda()
        sampled_unscaled_proposals = torch.gather(unscaled_proposals, dim=1, index=anchor_index)
        sampled_anchors = anchors[rpn_labels.anchor_ids]

        objectness_loss = F.binary_cross_entropy_with_logits(
            input=sampled_unscaled_proposals[:, :, 0],
            target=rpn_labels.objectness_labels.cuda(),
            reduction="mean",
        )
        bbox_loss = self.compute_regression_box_loss(
            unscaled_predicted_bboxes=sampled_unscaled_proposals[:, :, 1:],
            ground_truth_bboxes=rpn_labels.target_bboxes,
            reference_bboxes=sampled_anchors,
            weights=rpn_labels.objectness_labels,
        )

        return objectness_loss + self.bbox_loss_weight * bbox_loss

    @timed
    def select_rois(self, unscaled_proposals: torch.Tensor, anchors: torch.Tensor) -> List[torch.Tensor]:
        """
        :param unscaled_proposals: A tensor of B x R x (1 + 4) unscaled proposals.
        :param anchors: A tensor of R x 4 of anchors.
        :return: rois: A list of B tensors of size Pi x 4 proposals.
        """
        centered_anchors = center_bboxes(anchors)
        repeated_centered_anchors = torch.unsqueeze(centered_anchors, dim=0).repeat(self.num_images_per_batch, 1, 1)

        objectness_scores = unscaled_proposals[..., 0].cpu()
        print("unscaled proposals h/w", torch.max(unscaled_proposals[..., 3]), torch.max(unscaled_proposals[..., 4]))
        proposals = scale_bboxes(unscaled_proposals[..., 1:], repeated_centered_anchors.cuda()).cpu()
        print("scaled proposals h/w", torch.max(proposals[..., 2]), torch.max(proposals[..., 3]))

        max_rois_per_image = self.max_train_rois_per_image if self.training else self.max_eval_rois_per_image
        max_nms_candidates = self.max_train_nms_candidates if self.training else self.max_eval_nms_candidates
        selected_rois = []
        for i, image_proposals in enumerate(proposals):
            image_proposals = image_proposals[torch.argsort(objectness_scores[i], descending=True)[:max_nms_candidates]]

            image_rois = uncenter_bboxes(image_proposals)
            ious = compute_ious(image_rois, image_rois)
            selected_indices = [0]
            for j in range(1, max_nms_candidates):
                if torch.max(ious[j, selected_indices]).item() < self.nms_iou_threshold:
                    selected_indices.append(j)
                    if len(selected_indices) >= max_rois_per_image:
                        break

            selected_rois.append(image_rois[selected_indices])

        return selected_rois

    @timed
    def compute_detection_labels(
        self,
        rois: List[torch.Tensor],
        labels: List[List[Dict[str, Any]]],
    ) -> List[torch.Tensor]:
        """
        :param rois: A list of B tensors of size Pi x 4 .
        :param labels: A list of size B of ground truth annotation metadata.
        :return: A list of B tensors of size Pi x 5.
        """
        detection_labels = []

        for image_rois, image_labels in zip(rois, labels):
            gt_bboxes = parse_annotated_bboxes(image_labels)
            gt_categories = parse_annotated_categories(image_labels)
            ious = compute_ious(image_rois, gt_bboxes)

            best_ious, best_gts = torch.max(ious, dim=1)
            detection_labels.append(
                torch.cat(
                    [
                        torch.where(best_ious >= self.detection_iou_threshold, gt_categories[best_gts], 0).view(-1, 1),
                        gt_bboxes[best_gts]
                    ],
                    dim=1,
                )
            )

        return detection_labels

    @timed
    def compute_detections(
        self,
        feature_maps: List[torch.Tensor],
        rois: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param feature_maps: A list of L tensors of shape (B x C x H x W).
        :param rois: A list of B tensors of size Pi x 4 of rois.
        :return: A tuple of tensors:
            - classifications: a tensor of (B * P) * (K + 1) class predictions.
            - bboxes: a tensor of (B * P) x K x 4 predicted bboxes, one for each object class.
        """
        rois_by_level = [[] for _ in range(MAX_LEVEL)]
        perm_by_level = [[] for _ in range(MAX_LEVEL)]
        all_levels = []
        index = 0

        max_area = 0
        for image_index, image_rois in enumerate(rois):
            for roi in image_rois:
                # TODO(pauldb): Check the level distribution assignment.
                level = max(min(2 + int(math.log(math.sqrt(roi[-2] * roi[-1]) / 224, 2)), MAX_LEVEL-1), 0)
                max_area = max(max_area, roi[-2].item() * roi[-1].item())
                all_levels.append(level)
                rois_by_level[level].append([image_index, roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])
                perm_by_level[level].append(index)
                index += 1

        print(
            max_area,
            sum([len(level_rois) for level_rois in rois_by_level]),
            [(feature_map.size(), len(level_rois)) for feature_map, level_rois in zip(feature_maps, rois_by_level)]
        )

        start_time = time.time()
        pooled_feature_maps = []
        for level, (feature_map, level_rois) in enumerate(zip(feature_maps, rois_by_level)):
            level_start_time = time.time()
            pooled_feature_maps.append(
                O.roi_align(
                    feature_map,
                    torch.tensor(level_rois, device="cuda"),
                    output_size=7,
                    spatial_scale=1 / (4 * (1 << level)),
                    aligned=True,
                )
            )
            print(f"Level {level} took {time.time() - level_start_time}")

        end_time = time.time()
        print(f"Computing ROI align took {end_time - start_time}")

        start_time = end_time
        # Detection inference is performed by stacking level wise feature maps, but we must revert to the original
        # order in which the ground truth labels and ROIs (which act as anchors during detection) are defined.
        pooled_feature_maps = torch.cat(pooled_feature_maps, dim=0)
        perm = torch.cat([torch.tensor(x) for x in perm_by_level], dim=0)
        pooled_feature_maps[perm] = pooled_feature_maps.clone()

        detection_feature_maps = self.detection_trunk(pooled_feature_maps)
        classification_scores = self.classification_head(detection_feature_maps)
        bboxes = self.regression_head(detection_feature_maps).view(-1, self.num_classes, 4)

        end_time = time.time()
        print(f"Remaining inference took {end_time - start_time}")

        return classification_scores, bboxes

    @timed
    def compute_detection_loss(
        self,
        detected_class_logits: torch.Tensor,
        detected_bboxes: torch.Tensor,
        rois: List[torch.Tensor],
        detection_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        :param detected_class_logits: A tensor of size (B * P) x (K + 1).
        :param detected_bboxes: A tensor of size (B * P) X K x 4
        :param rois: A list of B tensors of size Pi x 4.
        :param detection_labels: A list of B tensors of size Pi x (1 + 4) representing ground truth detection labels.
        :return: A scalar tensor representing the loss.
        """
        rois = torch.cat(rois, dim=0)
        detection_labels = torch.cat(detection_labels, dim=0)

        class_labels = detection_labels[:, 0].to(dtype=torch.long, device="cuda")
        print(torch.min(class_labels), torch.max(class_labels))
        classification_loss = F.cross_entropy(detected_class_logits, class_labels)

        mask = class_labels > 0
        bbox_loss = self.compute_regression_box_loss(
            unscaled_predicted_bboxes=detected_bboxes[mask, class_labels[mask]-1, :],
            ground_truth_bboxes=detection_labels[mask, 1:],
            reference_bboxes=rois[mask],
        )
        return classification_loss + bbox_loss

    def forward(self, batch: Batch) -> Dict[str, Any]:
        """
        :param batch: A batch of training examples.
        """
        batch = batch.to(torch.device("cuda"))

        start_time = time.time()
        anchors = self.anchor_generator.generate(max_width=batch.images.shape[-1], max_height=batch.images.shape[-2])
        end_time = time.time()
        # print(f"Generating anchors took {end_time - start_time}")

        start_time = end_time
        feature_maps, unscaled_proposals = self.compute_unscaled_proposals(images=batch.images)
        end_time = time.time()
        # print(f"computing unscaled proposals took {end_time - start_time}")

        start_time = end_time
        rpn_labels = self.compute_proposal_labels(anchors=anchors, labels=batch.labels)
        end_time = time.time()
        # print(f"computing proposal labels took {end_time - start_time}")

        start_time = end_time
        proposal_loss = self.compute_rpn_loss(
            unscaled_proposals=unscaled_proposals,
            anchors=anchors,
            rpn_labels=rpn_labels,
        )
        end_time = time.time()
        # print(f"computing proposal loss took {end_time - start_time}")

        start_time = end_time
        rois = self.select_rois(unscaled_proposals=unscaled_proposals.detach(), anchors=anchors)
        end_time = time.time()
        # print(f"selecting rois took {end_time - start_time}")

        # start_time = end_time
        # detections_labels = self.compute_detection_labels(rois, batch.labels)
        # end_time = time.time()
        # # print(f"compute detection labels took {end_time - start_time}")
        #
        # start_time = end_time
        # class_logits, detected_bboxes = self.compute_detections(feature_maps, rois)
        # end_time = time.time()
        # print(f"compute detections took {end_time - start_time}")
        #
        # start_time = end_time
        # detection_loss = self.compute_detection_loss(class_logits, detected_bboxes, rois, detections_labels)
        # end_time = time.time()
        # # print(f"compute detection loss took {end_time - start_time}")

        return {
            "loss": proposal_loss, # + detection_loss,
            "time_stats": {k: np.mean(v) for k, v in self.time_stats.items()}
        }

    @timed
    def process_detections(
        self,
        rois: List[torch.Tensor],
        class_logits: torch.Tensor,
        detected_bboxes: torch.Tensor,
    ) -> Tuple[List[List[int]], List[List[List[float]]]]:
        """
        :param rois: A list of B tensors of size Pi x 4.
        :param class_logits: A tensor of shape (B * Pi) x (K + 1)
        :param detected_bboxes: A tensor of shape (B x Pi) x 4.
        :return: A tuple of detected_classes and detected_bboxes:
            - filtered_classes: A nested list of B x Pi elements representing the predicted non-background classes
            - filtered_bboxes: A nested list of size B x Pi x 4 representing the predicted bounding boxes.
        """
        image_detection_counts = [image_rois.shape[0] for image_rois in rois]
        detected_class_logits = torch.split(class_logits, split_size_or_sections=image_detection_counts, dim=0)
        detected_bboxes = torch.split(detected_bboxes, split_size_or_sections=image_detection_counts, dim=0)

        filtered_classes = []
        filtered_bboxes = []
        for image_class_logits, image_bboxes in zip(detected_class_logits, detected_bboxes):
            filtered_image_classes = []
            filtered_image_bboxes = []
            for class_logits, detected_bbox in zip(image_class_logits, image_bboxes):
                predicted_class = torch.argmax(class_logits)
                if predicted_class > 0:
                    filtered_image_classes.append(predicted_class.item())
                    filtered_image_bboxes.append(BBox.from_array(detected_bbox, centered=True).to_array(centered=False))
            filtered_classes.append(filtered_image_classes)
            filtered_bboxes.append(filtered_image_bboxes)

        return filtered_classes, filtered_bboxes

    def eval_forward(self, images: torch.Tensor) -> Dict[str, Any]:
        feature_maps, proposals = self.compute_proposals(images)
        rois = self.select_rois(proposals)
        class_logits, detected_bboxes = self.compute_detections(feature_maps, rois)
        detected_classes, detected_bboxes = self.process_detections(rois, class_logits, detected_bboxes)

        return {
            "classes": detected_classes,
            "bboxes": detected_bboxes,
        }
