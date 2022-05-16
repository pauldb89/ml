import functools
import itertools
import math
import time
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

from object_detection.bounding_box import BBox


def timed(fn: Callable) -> Callable:
    def wrapper(self: Detector, *args, **kwargs):
        start_time = time.time()
        ret = fn(self, *args, **kwargs)
        prefix = "train" if self.training else "eval"
        self.time_stats[f"{prefix}_{fn.__name__}"].append(time.time() - start_time)
        return ret
    return wrapper


MAX_LEVEL = 5
STRIDE_LEVEL_FACTOR = 4
ANCHOR_SIZE_LEVEL_FACTOR = 32


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


class AnchorIndex(NamedTuple):
    """
    Helper class to facilitate looking up anchors based on feature map locations.
    """
    # index of the image in the batch
    image_index: int
    # level in the feature map
    level: int
    # feature map index in the x-axis
    x_index: int
    # feature map index in the y-axis
    y_index: int
    # index of the aspect ratio
    aspect_index: int

    @property
    def feature_map_index(self) -> Tuple[int, int, int, int]:
        return self.image_index, self.x_index, self.y_index, self.aspect_index


class RPNLabels(NamedTuple):
    """
    Class to store label information for region proposals.

    Let R be the number of region proposals in a batch. Then:
    - objectness_labels: R - Whether a sampled region proposal matches a ground truth box.
    - anchor_bboxes: R x 4 - The bbox coordinates of the anchors for each sampled region proposal.
    - ground_truth_bboxes: R x 4 - The bbox coordinates for the ground truth bboxes of each sampled region proposal.
    - mask: R - Whether or not to include an image in the RPN loss calculation.
    - anchor_indexes: A list of R elements corresponding to the feature map locations for the sampled regions (anchors).
    """
    objectness_labels: torch.Tensor
    anchor_bboxes: torch.Tensor
    gt_bboxes: torch.Tensor
    mask: torch.Tensor
    anchor_indexes: List[AnchorIndex]


class Detector(nn.Module):
    def __init__(
        self,
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        rpn_positive_iou_threshold: float = 0.7,
        rpn_negative_iou_threshold: float = 0.3,
        num_images_per_batch: int = 2,
        rpn_regions_per_batch: int = 256,
        rpn_positive_regions_per_batch: int = 128,
        nms_iou_threshold: float = 0.7,
        max_train_rois_per_image: int = 2000,
        max_eval_rois_per_image: int = 1000,
        bbox_loss_weight: float = 10.0,
        detection_iou_threshold: float = 0.5,
        num_classes: int = 80,
    ):
        super().__init__()

        self.aspect_ratios = aspect_ratios
        self.rpn_positive_iou_threshold = rpn_positive_iou_threshold
        self.rpn_negative_iou_threshold = rpn_negative_iou_threshold
        self.num_images_per_batch = num_images_per_batch
        self.rpn_regions_per_batch = rpn_regions_per_batch
        self.rpn_positive_regions_per_batch = rpn_positive_regions_per_batch
        self.nms_iou_threshold = nms_iou_threshold
        self.max_train_rois_per_image = max_train_rois_per_image
        self.max_eval_rois_per_image = max_eval_rois_per_image
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
                padding=1,
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

    @timed
    def compute_proposal_labels(
        self,
        images: torch.Tensor,
        labels: List[List[Dict[str, Any]]],
    ) -> RPNLabels:
        """
        :param images: A tensor of size B x 3 x H x W.
        :param labels: A list of B ground truth labels representing classes and bounding boxes.
        :return: A tuple: proposal_labels and proposal_masks.
            - rpn_labels is a tensor of size B x L x H x W x A x (1 + 4)) representing the objectness score
              and proposal box for each image x layer x height x width x aspect ratio.
            - rpn_masks is a tensor of size B x L x H x W x A representing whether to use the proposal
              for image x layer x (height x width x aspect ratio) in the proposal loss.
        """
        image_height, image_width = images[0].shape[-2], images[0].shape[-1]

        anchors = []
        for level in range(MAX_LEVEL):
            stride = STRIDE_LEVEL_FACTOR * (1 << level)
            target_size = ANCHOR_SIZE_LEVEL_FACTOR * (1 << level)
            for b in range(len(labels)):
                for xi, x in enumerate(range(0, image_height, stride)):
                    for yi, y in enumerate(range(0, image_width, stride)):
                        for ai, aspect_ratio in enumerate(self.aspect_ratios):
                            anchor_index = AnchorIndex(
                                image_index=b,
                                level=level,
                                x_index=xi,
                                y_index=yi,
                                aspect_index=ai,
                            )
                            anchor_width = target_size * math.sqrt(aspect_ratio)
                            anchor_height = target_size / math.sqrt(aspect_ratio)
                            anchor_bbox = BBox(
                                x=max(0, x-anchor_width/2),
                                y=max(0, y-anchor_height/2),
                                w=min(image_width, anchor_width),
                                h=max(image_height, anchor_height),
                            )
                            anchors.append((anchor_index, anchor_bbox))

        anchor_max_iou = [0] * len(anchors)
        anchor_argmax_iou = [0] * len(anchors)
        gt_max_iou = [[0 for _ in label] for label in labels]
        gt_argmax_iou = [[() for _ in label] for label in labels]
        for i, (anchor_index, anchor_bbox) in enumerate(anchors):
            b = anchor_index.image_index
            for gti, annotation in enumerate(labels[b]):
                gt_bbox = BBox.from_array(annotation["bbox"], centered=False)

                iou = anchor_bbox.iou(gt_bbox)
                if iou > anchor_max_iou[i]:
                    anchor_max_iou[i], anchor_argmax_iou[i] = iou, gti

                if iou > gt_max_iou[b][gti]:
                    gt_max_iou[b][gti], gt_argmax_iou[b][gti] = iou, anchor_index

        positive_anchor_ids, negative_anchor_ids = set(itertools.chain(*gt_argmax_iou)), set()
        for anchor_id, max_iou in enumerate(anchor_max_iou):
            if max_iou >= self.rpn_positive_iou_threshold:
                positive_anchor_ids.add(anchor_id)
            elif max_iou <= self.rpn_negative_iou_threshold and anchor_id not in positive_anchor_ids:
                negative_anchor_ids.add(anchor_id)

        sampled_positive_anchor_ids = np.random.choice(
            list(positive_anchor_ids),
            size=min(len(positive_anchor_ids), self.rpn_positive_regions_per_batch),
            replace=False,
        )
        sampled_negative_anchor_ids = np.random.choice(
            list(negative_anchor_ids),
            size=min(len(negative_anchor_ids), self.rpn_regions_per_batch - len(positive_anchor_ids)),
            replace=False,
        )

        objectness_labels = torch.zeros(self.rpn_regions_per_batch)
        mask = torch.zeros(self.rpn_regions_per_batch)
        anchor_bboxes = torch.ones(self.rpn_regions_per_batch, 4)
        gt_bboxes = torch.ones(self.rpn_regions_per_batch, 4)
        anchor_indexes = []
        for i, anchor_id in enumerate(sampled_positive_anchor_ids.union(sampled_negative_anchor_ids)):
            anchor_index, anchor_bbox = anchors[anchor_id]
            anchor_indexes.append(anchor_index)
            objectness_labels[i] = anchor_id in positive_anchor_ids
            mask[i] = 1
            anchor_bboxes[i] = torch.tensor(anchor_bbox.to_array(centered=True))
            gt = labels[anchor_index.image_index][anchor_argmax_iou[anchor_id]]
            gt_bboxes[i] = torch.tensor(BBox.from_array(gt["bbox"], centered=False).to_array(centered=True))

        return RPNLabels(
            objectness_labels=objectness_labels,
            anchor_bboxes=anchor_bboxes,
            gt_bboxes=gt_bboxes,
            anchor_indexes=anchor_indexes,
            mask=mask,
        )

    @timed
    def compute_proposals(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        :param images: A tensor of images of size B x 3 x H x W.
        :return: A tuple: feature_maps and proposals.
            - feature_maps is a list of L tensors of size B x C x H x W representing the activations later used for ROI
              pooling.
            - proposals is a list of L tensors of size B x H x W x A x (1 + 4) representing region proposals.
        """
        self.backbone(images)

        feature_maps = [torch.zeros()] * len(self.tracked_feature_maps)
        for i, tracked_feature_map in reversed(list(enumerate(self.tracked_feature_maps))):
            feature_maps[i] = self.lateral_connections[i](tracked_feature_map)
            if i + 1 < len(self.tracked_feature_maps):
                feature_maps[i] += F.upsample(feature_maps[i+1], size=2, mode="nearest")
        feature_maps.append(F.interpolate(feature_maps[-1], scale_factor=2))

        proposals = []
        for i, feature_map in enumerate(feature_maps):
            proposal = self.region_proposal_head(self.feature_map_transforms[i](feature_map))
            # Shape B x (A * (1 + 4)) x H x W ==> B x H x W x (A * (4 + 1)).
            proposal = torch.permute(proposal, dims=[0, 2, 3, 1])
            # Shape B x H x W x (A * (4 + 1)) ==> B x H x W x A x (4 + 1)
            proposals.append(proposal.view(*proposal.shape[:3], len(self.aspect_ratios), 5))

        return feature_maps, proposals

    @timed
    def compute_regression_box_loss(
        self,
        predicted_bboxes: torch.Tensor,
        ground_truth_bboxes: torch.Tensor,
        reference_bboxes: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """
        :param predicted_bboxes: N x 4 tensor representing the centered bbox coordinates of the predictions.
        :param ground_truth_bboxes: N x 4 tensor representing the centered bbox coordinates of the ground truth labels.
        :param reference_bboxes: N x 4 tensor representing the centered coordinates of the reference bboxes.
        :param weights: N tensors representing individual weights (or mask).
        :return:
        """
        location_loss = F.smooth_l1_loss(
            (predicted_bboxes[:, 0:2] - reference_bboxes[:, 0:2]) / reference_bboxes[:, 2:4],
            (ground_truth_bboxes[:, 0:2] - reference_bboxes[:, 0:2]) / reference_bboxes[:, 2:4],
            reduction="none",
        )
        size_loss = F.smooth_l1_loss(
            torch.log(predicted_bboxes[:, 2:4] / reference_bboxes[:, 2:4]),
            torch.log(ground_truth_bboxes[:, 2:4] / reference_bboxes[:, 2:4]),
            reduction="none",
        )
        bbox_loss = torch.mean(
            torch.cat([location_loss, size_loss], dim=1).mean(dim=1) * weights
        )
        return bbox_loss

    @timed
    def compute_proposals_loss(self, proposals: List[torch.Tensor], rpn_labels: RPNLabels) -> torch.Tensor:
        """
        :param proposals: A tensor of region proposals of size B x L x H x W x A x (1 + 4).
        :param rpn_labels: A RPNLabels struct containing a sampled set of labels for computing the loss.
        :return: A scalar tensor representing the loss.
        """
        predictions = []
        predicted_bboxes = []
        for anchor_index in rpn_labels.anchor_indexes:
            proposal = proposals[anchor_index.level][anchor_index.feature_map_index]
            predictions.append(proposal[0])
            predicted_bboxes.append(proposal[1:])

        predictions = torch.stack(predictions, dim=0)
        predicted_bboxes = torch.stack(predicted_bboxes, dim=0)

        objectness_loss = F.binary_cross_entropy_with_logits(
            input=predictions,
            target=rpn_labels.objectness_labels,
            weight=rpn_labels.mask,
            reduction="mean",
        )
        bbox_loss = self.compute_regression_box_loss(
            predicted_bboxes=predicted_bboxes,
            ground_truth_bboxes=rpn_labels.gt_bboxes,
            reference_bboxes=rpn_labels.anchor_bboxes,
            weights=rpn_labels.mask * rpn_labels.objectness_labels,
        )

        return objectness_loss + self.bbox_loss_weight * bbox_loss

    @timed
    def select_rois(self, input_proposals: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        :param input_proposals: A list of L tensor of dense region proposals of size B x H x W x A x (1 + 4)).
        :return: rois: A list of tensors of size Pi x 4 proposals.
        """
        max_rois_per_image = self.max_train_rois_per_image if self.training else self.max_eval_rois_per_image
        image_proposals = [[] for _ in range(self.num_images_per_batch)]
        for level_proposals in input_proposals:
            for i in range(self.num_images_per_batch):
                p = level_proposals[i].view(-1, 5)
                scores = p[:, 0]
                bboxes = p[:, 1:]
                image_proposals[i].extend(
                    (score, BBox.from_array(bbox, centered=True)) for score, bbox in zip(scores, bboxes)
                )

        rois = [[] for _ in range(self.num_images_per_batch)]
        for i, proposals in enumerate(sorted(image_proposals, reverse=True)):
            for _, bbox in proposals:
                valid = True
                for selected_bbox in rois:
                    if bbox.iou(selected_bbox) >= self.nms_iou_threshold:
                        valid = False
                        break

                if valid:
                    rois.append(bbox)
                    if len(rois) >= max_rois_per_image:
                        break

        return [
            torch.stack([bbox.to_array(centered=True) for bbox in image_rois], dim=0)
            for image_rois in rois
        ]

    @timed
    def compute_detection_labels(
        self,
        rois: List[torch.Tensor],
        labels: List[List[Dict[str, Any]]],
    ) -> torch.Tensor:
        """
        :param rois: A list of B tensors of size Pi x 4 .
        :param labels: A list of size B of ground truth annotation metadata.
        :return: A tensor of size (B x P) x (1 + 4) labels.
        """
        detection_labels = []

        for i, image_rois in enumerate(rois):
            for roi in image_rois:
                roi_bbox = BBox.from_array(roi, centered=False)
                max_iou = 0
                argmax_iou = 0
                for j, annotation in enumerate(labels[i]):
                    gt_bbox = BBox.from_array(annotation["bbox"], centered=False)
                    iou = roi_bbox.roi(gt_bbox)
                    if iou >= max_iou:
                        max_iou = iou
                        argmax_iou = j

                if max_iou >= self.detection_iou_threshold:
                    gt_detection = labels[i][argmax_iou]
                    detection_labels.append(
                        torch.tensor([
                            gt_detection["category_id"],
                            *BBox.from_array(gt_detection["bbox"], centered=False).to_array(centered=True)
                        ])
                    )
                else:
                    detection_labels.append(torch.tensor([0, 1, 1, 1, 1], dtype=torch.float))

        return torch.stack(detection_labels, dim=0)

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
        index = 0
        for image_rois in rois:
            for roi in image_rois:
                # TODO(pauldb): Check the level distribution assignment.
                level = max(min(2 + int(math.log(math.sqrt(roi[-2] * roi[-1]) / 224, base=2)), MAX_LEVEL-1), 0)
                rois_by_level[level].append(roi)
                perm_by_level[level].append(index)
                index += 1

        pooled_feature_maps = []
        for level, (feature_map, level_rois) in enumerate(zip(feature_maps, rois_by_level)):
            pooled_feature_maps.append(
                O.roi_align(
                    feature_map,
                    level_rois,
                    output_size=7,
                    spatial_scale=1 / (4 * (1 << level)),
                    aligned=True,
                )
            )

        # Detection inference is performed by stacking level wise feature maps, but we must revert to the original
        # order in which the ground truth labels and ROIs (which act as anchors during detection) are defined.
        detection_feature_maps = self.detection_trunk(torch.cat(pooled_feature_maps, dim=0))
        perm = torch.cat([torch.tensor(x) for x in perm_by_level], dim=0)
        detection_feature_maps[perm] = detection_feature_maps.clone()

        classification_scores = self.classification_head(detection_feature_maps)
        bboxes = self.regression_head(detection_feature_maps).view(-1, self.num_classes, 4)
        return classification_scores, bboxes

    @timed
    def compute_detection_loss(
        self,
        detected_class_logits: torch.Tensor,
        detected_bboxes: torch.Tensor,
        rois: List[torch.Tensor],
        detection_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param detected_class_logits: A tensor of size (B * P).
        :param detected_bboxes: A tensor of size (B * P) X K x 4
        :param rois: A list of B tensors of size Pi x 4.
        :param detection_labels: A tensor of size (B * P) x (1 + 4) representing ground truth detection labels.
        :return: A scalar tensor representing the loss.
        """
        class_labels = detection_labels[:, 0]
        classification_loss = F.cross_entropy(detected_class_logits, class_labels)

        mask = class_labels > 0
        bbox_loss = self.compute_regression_box_loss(
            predicted_bboxes=detected_bboxes[mask, class_labels[mask]-1, :],
            ground_truth_bboxes=detection_labels[mask, 1:],
            reference_bboxes=torch.cat(rois, dim=0),
        )
        return classification_loss + bbox_loss

    def forward(self, images: torch.Tensor, labels: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        :param images: A list of images of size B x 3 x H x W representing a batch of images.
        :param labels: A list of B metadata entries representing ground truth classes and bounding boxes.
        :return:
        """
        feature_maps, proposals = self.compute_proposals(images)
        rpn_labels = self.compute_proposal_labels(images, labels)
        proposal_loss = self.compute_proposals_loss(proposals, rpn_labels)

        rois = self.select_rois(proposals)

        detections_labels = self.compute_detection_labels(rois, labels)
        class_logits, detected_bboxes = self.compute_detections(feature_maps, rois)
        detection_loss = self.compute_detection_loss(class_logits, detected_bboxes, rois, detections_labels)

        return {
            "loss": proposal_loss + detection_loss,
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
