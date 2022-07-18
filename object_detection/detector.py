import math
import time
from collections import defaultdict
from collections import deque
from typing import Any, Final
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torchvision.models
import torchvision.ops as O
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet
from torchvision.ops import batched_nms

from object_detection.anchors import AnchorGenerator
from object_detection.anchors import sample_labels
from object_detection.data import Batch, Label, Labels

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
Ri - Total number of anchors / region proposals for an image = A * H_i * W_i
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


def to_xxyy(bboxes: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [bboxes[..., 0], bboxes[..., 1], bboxes[..., 0] + bboxes[..., 2], bboxes[..., 1] + bboxes[..., 3]],
        dim=-1,
    )


def scale_bboxes(
    bboxes: torch.Tensor,
    centered_anchors: torch.Tensor,
    weights: torch.Tensor,
    max_scale: float = math.log(1000.0 / 16),
) -> torch.Tensor:
    assert bboxes.shape == centered_anchors.shape, (bboxes.shape, centered_anchors.shape)

    x = (bboxes[..., 0] / weights[0]) * centered_anchors[..., 2] + centered_anchors[..., 0]
    y = (bboxes[..., 1] / weights[1]) * centered_anchors[..., 3] + centered_anchors[..., 1]
    w = centered_anchors[..., 2] * torch.exp(torch.clip(bboxes[..., 2] / weights[2], max=max_scale))
    h = centered_anchors[..., 3] * torch.exp(torch.clip(bboxes[..., 3] / weights[3], max=max_scale))

    return uncenter_bboxes(torch.stack([x, y, w, h], dim=-1))


def unscale_bboxes(bboxes: torch.Tensor, anchors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    x = (bboxes[..., 0] - anchors[..., 0]) / anchors[..., 2] * weights[0]
    y = (bboxes[..., 1] - anchors[..., 1]) / anchors[..., 3] * weights[1]
    w = torch.log(bboxes[..., 2] / anchors[..., 2]) * weights[2]
    h = torch.log(bboxes[..., 3] / anchors[..., 3]) * weights[3]
    return torch.stack([x, y, w, h], dim=-1)


def clip_bboxes(bboxes: torch.Tensor, max_width: int, max_height: int) -> torch.Tensor:
    x1 = torch.clip(bboxes[..., 0], min=0, max=max_width)
    y1 = torch.clip(bboxes[..., 1], min=0, max=max_height)
    x2 = torch.clip(bboxes[..., 0] + bboxes[..., 2], min=0, max=max_width)
    y2 = torch.clip(bboxes[..., 1] + bboxes[..., 3], min=0, max=max_height)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def parse_annotated_bboxes(label: Label) -> torch.Tensor:
    bboxes = []
    for annotation in label["annotations"]:
        bbox = annotation["bbox"]
        if bbox[2] > 0 and bbox[3] > 0:
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
    return torch.tensor(bboxes, device="cuda")


def parse_annotated_categories(label: Label) -> torch.Tensor:
    return torch.tensor([annotation["category_id"] for annotation in label["annotations"]], device="cuda")


class ResNetWrapper(nn.Module):
    def __init__(self, resnet: ResNet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)

        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        ret = []
        x = self.resnet.layer1(x)
        ret.append(x)
        x = self.resnet.layer2(x)
        ret.append(x)
        x = self.resnet.layer3(x)
        ret.append(x)
        x = self.resnet.layer4(x)
        ret.append(x)

        return ret


class Detector(nn.Module):
    RPN_CLS_LOSS: Final[str] = "loss_rpn_cls"
    RPN_BOX_LOSS: Final[str] = "loss_rpn_loc"
    DETECTION_CLS_LOSS: Final[str] = "loss_cls"
    DETECTION_BOX_LOSS: Final[str] = "loss_box_reg"
    LOSS_KEYS: Final[Tuple[str, ...]] = [DETECTION_CLS_LOSS, DETECTION_BOX_LOSS, RPN_CLS_LOSS, RPN_BOX_LOSS]

    def __init__(
        self,
        resolutions: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (4, 8, 16, 32, 64),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        rpn_positive_iou_threshold: float = 0.7,
        rpn_negative_iou_threshold: float = 0.3,
        rpn_regions_per_batch: int = 256,
        rpn_positive_regions_fraction: float = 0.5,
        rpn_objectness_loss_weight: float = 1.0,
        rpn_bbox_loss_weight: float = 1.0,
        rpn_bbox_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        rpn_nms_iou_threshold: float = 0.7,
        max_train_nms_candidates: int = 2000,
        max_eval_nms_candidates: int = 1000,
        max_train_rois_per_image: int = 1000,
        max_eval_rois_per_image: int = 1000,
        detection_positive_iou_threshold: float = 0.5,
        detection_nms_iou_threshold: float = 0.5,
        detections_per_batch: int = 512,
        detections_positive_fraction: float = 0.25,
        num_detection_layers: int = 4,
        detection_bbox_weights: Sequence[float] = (10.0, 10.0, 5.0, 5.0),
        detection_class_score_threshold: float = 0.05,
        max_detections: int = 100,
        smooth_l1_beta: float = 0.0,
        num_classes: int = 80,
    ):
        super().__init__()

        self.resolutions = resolutions
        self.aspect_ratios = aspect_ratios
        self.anchor_generator = AnchorGenerator(resolutions=resolutions, strides=strides, aspect_ratios=aspect_ratios)

        self.rpn_positive_iou_threshold = rpn_positive_iou_threshold
        self.rpn_negative_iou_threshold = rpn_negative_iou_threshold
        self.rpn_regions_per_batch = rpn_regions_per_batch
        self.rpn_positive_regions_fraction = rpn_positive_regions_fraction
        self.rpn_objectness_loss_weight = rpn_objectness_loss_weight
        self.rpn_bbox_loss_weight = rpn_bbox_loss_weight
        self.rpn_nms_iou_threshold = rpn_nms_iou_threshold
        self.max_train_rois_per_image = max_train_rois_per_image
        self.max_eval_rois_per_image = max_eval_rois_per_image
        self.max_train_nms_candidates = max_train_nms_candidates
        self.max_eval_nms_candidates = max_eval_nms_candidates
        self.detection_positive_iou_threshold = detection_positive_iou_threshold
        self.detection_nms_iou_threshold = detection_nms_iou_threshold
        self.detections_per_batch = detections_per_batch
        self.detections_positive_fraction = detections_positive_fraction
        self.num_detection_layers = num_detection_layers
        self.detection_class_score_threshold = detection_class_score_threshold
        self.max_detections = max_detections
        self.smooth_l1_beta = smooth_l1_beta
        self.num_classes = num_classes

        self.register_buffer("rpn_bbox_weights", torch.tensor(rpn_bbox_weights))
        self.register_buffer("detection_bbox_weights", torch.tensor(detection_bbox_weights))

        self.time_stats = defaultdict(lambda: deque(maxlen=100))

        self.backbone = ResNetWrapper(torchvision.models.resnet50(pretrained=True, norm_layer=O.FrozenBatchNorm2d))

        self.lateral_connections = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for in_channels in [256, 512, 1024, 2048]
        ])
        # Conv transforms which help reduce the aliasing effect of up-sampling. Interesting
        # that they are not added directly on the top-down path.
        self.feature_map_transforms = nn.ModuleList([
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for _ in range(4)
        ])

        self.region_proposal_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=5 * len(self.aspect_ratios),
                kernel_size=1,
                stride=1,
                padding=0,
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

    def compute_unscaled_proposals(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        :param images: A tensor of images of size B x 3 x H x W.
        :return: A tuple:
            - feature_maps: A list of L tensors of shape B x C x Hi x Wi
            - unscaled_proposals: A list of L tensors of shape B x Ri x (1 + 4).
        """
        backbone_feature_maps = self.backbone(images)

        feature_maps = [torch.tensor(0) for _ in range(len(backbone_feature_maps))]
        for i, tracked_feature_map in reversed(list(enumerate(backbone_feature_maps))):
            x = self.lateral_connections[i](tracked_feature_map)
            if i+1 < len(feature_maps):
                feature_maps[i] = x + F.interpolate(feature_maps[i+1], scale_factor=2, mode="nearest")
            else:
                feature_maps[i] = x

        for i, transform in enumerate(self.feature_map_transforms):
            feature_maps[i] = transform(feature_maps[i])
        feature_maps.append(F.max_pool2d(feature_maps[-1], kernel_size=1, stride=2, padding=0))

        unscaled_proposals = []
        for i, feature_map in enumerate(feature_maps):
            proposal = self.region_proposal_head(feature_map)
            # Shape B x (A * (1 + 4)) x H x W ==> B x H x W x (A * (1 + 4)).
            proposal = torch.permute(proposal, dims=[0, 2, 3, 1])
            # Shape B x H x W x (A * (1 + 4)) ==> B x (H * W * A) x (1 + 4)
            unscaled_proposals.append(proposal.reshape(images.shape[0], -1, 5))

        return feature_maps, unscaled_proposals

    @timed
    def compute_proposal_labels(self, anchors: List[torch.Tensor], labels: Labels) -> RPNLabels:
        """
        :param anchors: A list of L tensors of size Ri x 4.
        :param labels: A list of size B containing lists with the ground truth annotations for each image.
        :return: RPN labels.
        """
        all_anchors = torch.cat(anchors, dim=0)

        objectness_labels = []
        anchor_ids = []
        target_bboxes = []
        for image_labels in labels:
            gt_bboxes = parse_annotated_bboxes(image_labels)

            ious = compute_ious(all_anchors, gt_bboxes)

            best_ious, best_gts = ious.max(dim=1)
            positive_mask = best_ious >= self.rpn_positive_iou_threshold

            # For each ground truth box, mark all anchors with highest overlap as positives (including all ties!).
            best_gt_ious, _ = ious.max(dim=0)
            positive_mask[torch.any(ious >= best_gt_ious, dim=1)] = 1

            negative_mask = torch.logical_and(
                torch.logical_not(positive_mask),
                best_ious < self.rpn_negative_iou_threshold,
            )

            anchor_mask = torch.full_like(best_ious, -1, dtype=torch.int8)
            anchor_mask[positive_mask] = 1
            anchor_mask[negative_mask] = 0

            positive_anchor_ids, negative_anchor_ids = sample_labels(
                mask=anchor_mask,
                positive_fraction=self.rpn_positive_regions_fraction,
                num_samples=self.rpn_regions_per_batch,
                negative_class=0,
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
        bbox_weights: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """
        In the following X is either B x S (for the RPN loss) or P (for the detection bounding box loss).
        :param unscaled_predicted_bboxes: X x 4 tensor representing the centered bbox coordinates of the predictions.
        :param ground_truth_bboxes: X x 4 tensor representing the bbox coordinates of the ground truth labels.
        :param reference_bboxes: X x 4 tensor representing the coordinates of the reference bboxes.
        :param unscale_weights: 1D tensor containing 4 weights representing how much the reference boxes should be
                                scaled in each dimension (x, y, w, h).
        :param weights: X tensors representing individual weights (or mask).
        :return:
        """
        unscaled_predicted_bboxes = unscaled_predicted_bboxes.cuda()
        ground_truth_bboxes = ground_truth_bboxes.cuda()
        reference_bboxes = reference_bboxes.cuda()
        weights = weights.cuda() if isinstance(weights, torch.Tensor) else weights

        ground_truth_bboxes = center_bboxes(ground_truth_bboxes)
        reference_bboxes = center_bboxes(reference_bboxes)
        target_bboxes = unscale_bboxes(ground_truth_bboxes, reference_bboxes, weights=bbox_weights)

        losses = F.smooth_l1_loss(
            unscaled_predicted_bboxes,
            target_bboxes,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        return torch.mean(losses.sum(dim=-1) * weights)

    @timed
    def compute_rpn_loss(
        self,
        unscaled_proposals: List[torch.Tensor],
        anchors: List[torch.Tensor],
        rpn_labels: RPNLabels,
    ) -> Dict[str, torch.Tensor]:
        """
        :param unscaled_proposals: A list of L tensors of region proposals of size B x Ri x (1 + 4).
        :param anchors: A list of L tensors of anchors of size Ri x 4.
        :param rpn_labels: A RPNLabels struct containing a sampled set of labels for computing the loss.
        :return: A scalar tensor representing the loss.
        """
        anchors = torch.cat(anchors, dim=0)
        unscaled_proposals = torch.cat(unscaled_proposals, dim=1)

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
            bbox_weights=self.rpn_bbox_weights,
            weights=rpn_labels.objectness_labels,
        )

        return {
            self.RPN_CLS_LOSS: self.rpn_objectness_loss_weight * objectness_loss,
            self.RPN_BOX_LOSS: self.rpn_bbox_loss_weight * bbox_loss
        }

    @timed
    def select_rois(
        self,
        batch: Batch,
        unscaled_proposals: List[torch.Tensor],
        anchors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        :param batch: Current batch of examples (necessary for clipping ROIs to the original image sizes).
        :param unscaled_proposals: A list of L tensors of B x Ri x (1 + 4) unscaled proposals.
        :param anchors: A list of L tensors of Ri x 4 of anchors.
        :return: rois: A list of B tensors of size Pi x 4 proposals.
        """
        num_images_per_batch = len(batch.image_sizes)

        proposals = []
        objectness_scores = []
        for anchors_i, unscaled_proposals_i in zip(anchors, unscaled_proposals):
            centered_anchors = center_bboxes(anchors_i)
            repeated_centered_anchors = torch.unsqueeze(centered_anchors, dim=0).repeat(num_images_per_batch, 1, 1)

            objectness_scores.append(unscaled_proposals_i[..., 0])
            scaled_proposals = scale_bboxes(
                bboxes=unscaled_proposals_i[..., 1:],
                centered_anchors=repeated_centered_anchors.cuda(),
                weights=self.rpn_bbox_weights,
            )
            proposals.append(scaled_proposals)

        max_rois_per_image = self.max_train_rois_per_image if self.training else self.max_eval_rois_per_image
        max_nms_candidates = self.max_train_nms_candidates if self.training else self.max_eval_nms_candidates

        roi_scores = []
        roi_candidates = []
        candidate_levels = []
        for level in range(len(proposals)):
            k = min(max_nms_candidates, objectness_scores[level].shape[1])
            scores, indices = torch.topk(objectness_scores[level], k=k, dim=1)
            roi_scores.append(scores)
            repeated_indices = torch.unsqueeze(indices, dim=2).repeat(1, 1, 4)
            roi_candidates.append(torch.gather(proposals[level], dim=1, index=repeated_indices))
            candidate_levels.append(torch.full((k,), level))

        roi_scores = torch.cat(roi_scores, dim=1)
        roi_candidates = torch.cat(roi_candidates, dim=1)
        candidate_levels = torch.cat(candidate_levels, dim=0)

        selected_rois = []
        for image_id in range(roi_candidates.shape[0]):
            roi_candidates[image_id] = clip_bboxes(
                roi_candidates[image_id],
                max_width=batch.image_sizes[image_id][0],
                max_height=batch.image_sizes[image_id][1],
            )
            indices = batched_nms(
                boxes=to_xxyy(roi_candidates[image_id]),
                scores=roi_scores[image_id],
                idxs=candidate_levels,
                iou_threshold=self.rpn_nms_iou_threshold,
            )
            indices = indices[:max_rois_per_image]

            selected_image_rois = roi_candidates[image_id][indices]

            if self.training:
                # Add ground truth proposals as ROIs. This is to accelerate training in the first iterations when
                # the RPN hasn't learned how to produce high quality ROIs yet.
                gt_bboxes = parse_annotated_bboxes(batch.labels[image_id])
                selected_image_rois = torch.cat([selected_image_rois, gt_bboxes], dim=0)

            selected_rois.append(selected_image_rois)

        return selected_rois

    @timed
    def compute_detection_labels(
        self,
        rois: List[torch.Tensor],
        labels: Labels,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        :param rois: A list of B tensors of size Pi x 4 .
        :param labels: A list of size B of ground truth annotation metadata.
        :return: A tuple containing:
            sampled_rois: A list of B tensors of size P'i x 4
            detection_labels: A list of B tensors of size P'i x 5.
        """
        sampled_rois = []
        detection_labels = []

        for image_rois, image_labels in zip(rois, labels):
            # image_rois = torch.stack(sorted(image_rois, key=lambda row: row.cpu().numpy().tolist()))
            gt_bboxes = parse_annotated_bboxes(image_labels)
            gt_categories = parse_annotated_categories(image_labels)
            ious = compute_ious(image_rois, gt_bboxes)

            best_ious, best_gts = torch.max(ious, dim=1)

            label_mask = torch.where(
                best_ious >= self.detection_positive_iou_threshold,
                gt_categories[best_gts],
                self.num_classes,
            )

            positive_ids, negative_ids = sample_labels(
                mask=label_mask,
                positive_fraction=self.detections_positive_fraction,
                num_samples=self.detections_per_batch,
                negative_class=self.num_classes,
            )

            all_ids = torch.cat([positive_ids, negative_ids], dim=-1)
            sampled_rois.append(image_rois[all_ids])
            detection_labels.append(torch.cat([label_mask[all_ids].view(-1, 1), gt_bboxes[best_gts[all_ids]]], dim=1))

        return sampled_rois, detection_labels

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
        all_levels = []
        image_ids = []
        for image_id, image_rois in enumerate(rois):
            areas = image_rois[:, -1] * image_rois[:, -2]
            levels = 2 + torch.floor(torch.log2(torch.sqrt(areas) / 224 + 1e-6))
            levels = torch.clip(levels, min=0, max=self.num_detection_layers-1)
            all_levels.append(levels)
            image_ids.append(torch.full_like(levels, image_id))

        rois = torch.cat(rois, dim=0)
        all_levels = torch.cat(all_levels, dim=0)
        image_ids = torch.cat(image_ids, dim=0)

        pooled_feature_maps = []
        perm = []
        for level, feature_map in enumerate(feature_maps):
            mask = all_levels == level
            perm.append(mask.nonzero().view(-1))
            indexed_rois = torch.cat([image_ids[mask].view(-1, 1), to_xxyy(rois[mask])], dim=1)
            pooled_feature_maps.append(
                O.roi_align(
                    feature_map,
                    indexed_rois,
                    output_size=7,
                    spatial_scale=1 / (4 * (1 << level)),
                    aligned=True,
                )
            )

        # Detection inference is performed by stacking level wise feature maps, but we must revert to the original
        # order in which the ground truth labels and ROIs (which act as anchors during detection) are defined.
        pooled_feature_maps = torch.cat(pooled_feature_maps, dim=0)
        perm = torch.cat(perm, dim=0)
        pooled_feature_maps[perm] = pooled_feature_maps.clone()

        detection_feature_maps = self.detection_trunk(pooled_feature_maps)

        classification_scores = self.classification_head(detection_feature_maps)
        bboxes = self.regression_head(detection_feature_maps).view(-1, self.num_classes, 4)

        return classification_scores, bboxes

    @timed
    def compute_detection_loss(
        self,
        detected_class_logits: torch.Tensor,
        detected_bboxes: torch.Tensor,
        rois: List[torch.Tensor],
        detection_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
        classification_loss = F.cross_entropy(detected_class_logits, class_labels)

        mask = class_labels != self.num_classes
        bbox_loss = self.compute_regression_box_loss(
            unscaled_predicted_bboxes=detected_bboxes[mask, class_labels[mask], :],
            ground_truth_bboxes=detection_labels[mask, 1:],
            reference_bboxes=rois[mask],
            bbox_weights=self.detection_bbox_weights,
            weights=mask.sum() / class_labels.numel(),
        )

        return {
            self.DETECTION_CLS_LOSS: classification_loss,
            self.DETECTION_BOX_LOSS: bbox_loss
        }

    def forward(self, batch: Batch) -> Dict[str, Any]:
        """
        :param batch: A batch of training examples.
        """
        batch = batch.to(torch.device("cuda"))

        anchors = self.anchor_generator(max_width=batch.images.shape[-1], max_height=batch.images.shape[-2])

        feature_maps, unscaled_proposals = self.compute_unscaled_proposals(images=batch.images)

        rpn_labels = self.compute_proposal_labels(anchors=anchors, labels=batch.labels)

        losses = self.compute_rpn_loss(
            unscaled_proposals=unscaled_proposals,
            anchors=anchors,
            rpn_labels=rpn_labels,
        )

        with torch.no_grad():
            rois = self.select_rois(
                batch=batch,
                unscaled_proposals=unscaled_proposals,
                anchors=anchors,
            )

        sampled_rois, detections_labels = self.compute_detection_labels(rois, batch.labels)

        class_logits, detected_bboxes = self.compute_detections(feature_maps, sampled_rois)

        losses.update(self.compute_detection_loss(class_logits, detected_bboxes, sampled_rois, detections_labels))

        return {
            **losses,
            "loss": sum([losses[key] for key in self.LOSS_KEYS]),
        }

    @timed
    def process_detections(
        self,
        batch: Batch,
        rois: List[torch.Tensor],
        class_logits: torch.Tensor,
        detected_bboxes: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        :param batch: The input batch.
        :param rois: A list of B tensors of size Pi x 4.
        :param class_logits: A tensor of shape (B * Pi) x (K + 1)
        :param detected_bboxes: A tensor of shape (B x Pi) x K x 4.
        :return: A tuple of detected_classes and detected_bboxes:
            - filtered_classes: A nested list of B x Pi elements representing the predicted non-background classes
            - filtered_bboxes: A nested list of size B x Pi x 4 representing the predicted bounding boxes.
        """

        image_detection_counts = [image_rois.shape[0] for image_rois in rois]
        detected_class_logits = torch.split(class_logits, split_size_or_sections=image_detection_counts, dim=0)
        detected_bboxes = torch.split(detected_bboxes, split_size_or_sections=image_detection_counts, dim=0)

        output_classes = []
        output_bboxes = []
        output_scores = []
        for i in range(len(rois)):
            image_rois = center_bboxes(rois[i])
            repeated_rois = torch.unsqueeze(image_rois, dim=1).repeat(1, self.num_classes, 1).cuda()
            scaled_bboxes = scale_bboxes(
                bboxes=detected_bboxes[i],
                centered_anchors=repeated_rois,
                weights=self.detection_bbox_weights
            )

            scaled_bboxes = clip_bboxes(
                scaled_bboxes,
                max_width=batch.image_sizes[i][0],
                max_height=batch.image_sizes[i][1],
            )

            class_scores = F.softmax(detected_class_logits[i], dim=-1)[:, :self.num_classes]

            mask = class_scores > self.detection_class_score_threshold
            filtered_class_scores = class_scores[mask]
            filtered_scaled_bboxes = scaled_bboxes[mask]
            filtered_classes = mask.nonzero()[:, 1]

            indices = batched_nms(
                boxes=to_xxyy(filtered_scaled_bboxes).view(-1, 4),
                scores=filtered_class_scores.view(-1),
                idxs=filtered_classes,
                iou_threshold=self.detection_nms_iou_threshold,
            )[:self.max_detections]

            output_classes.append(filtered_classes[indices])
            output_bboxes.append(filtered_scaled_bboxes[indices])
            output_scores.append(filtered_class_scores[indices])

        return {
            "bboxes": output_bboxes,
            "classes": output_classes,
            "scores": output_scores,
        }

    def eval_forward(self, batch: Batch) -> Dict[str, Any]:
        batch = batch.to(device=torch.device("cuda"))

        anchors = self.anchor_generator(max_width=batch.images.shape[-1], max_height=batch.images.shape[-2])

        feature_maps, unscaled_proposals = self.compute_unscaled_proposals(batch.images)

        rois = self.select_rois(batch=batch, unscaled_proposals=unscaled_proposals, anchors=anchors)

        class_logits, detected_bboxes = self.compute_detections(feature_maps, rois)

        return self.process_detections(
            batch=batch,
            rois=rois,
            class_logits=class_logits,
            detected_bboxes=detected_bboxes,
        )

    def summarize(self, step: int, epoch: int, metrics: Dict[str, Any]) -> None:
        formatted_metrics = "\t".join([f"{k}: {metrics[k].item()}" for k in self.LOSS_KEYS])
        formatted_perf = "\t".join([f"{k}: {np.mean(v):.3f}" for k, v in self.time_stats.items()])
        print(f"Step {step}:\t{formatted_metrics}")
        print(f"Step {step}:\t{formatted_perf}")
