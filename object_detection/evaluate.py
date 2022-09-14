import itertools
import time
from typing import List

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader

from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size
from common.consts.coco_consts import EVAL_ANNOTATION_FILE
from object_detection.data import Batch
from object_detection.detector import Detector


def rescale_bbox(bbox: List[float], x_scale: float, y_scale: float) -> List[float]:
    return [bbox[0] * x_scale, bbox[1] * y_scale, bbox[2] * x_scale, bbox[3] * y_scale]


def evaluate(step: int, model: Detector, data_loader: DataLoader) -> None:
    print_once(f"Running inference on eval dataset at step {step}")
    start_time = time.time()
    category_mapping = data_loader.dataset.categories

    detections = []
    with torch.no_grad():
        model.eval()

        for batch_id, batch in enumerate(data_loader, start=1):
            num_images = len(batch.image_sizes)
            if batch_id % 10 == 0:
                print_once(f"Finished evaluating {world_size() * batch_id * num_images} examples")

            result = model.eval_forward(Batch(images=batch.images, labels=None, image_sizes=batch.image_sizes))
            for i in range(num_images):
                labels = batch.labels[i]
                image_id = labels["image_id"]
                original_width = labels["original_width"]
                original_height = labels["original_height"]
                width, height = batch.image_sizes[i]

                image_classes = result["classes"][i].cpu().numpy().tolist()
                image_bboxes = result["bboxes"][i].cpu().numpy().tolist()
                image_scores = result["scores"][i].cpu().numpy().tolist()

                new_detections = []
                for predicted_class, bbox, score in zip(image_classes, image_bboxes, image_scores):
                    new_detections.append({
                        "category_id": category_mapping[predicted_class],
                        "bbox": rescale_bbox(bbox, x_scale=original_width / width, y_scale=original_height / height),
                        "image_id": image_id,
                        "id": len(detections),
                        "score": score,
                    })

                detections.extend(new_detections)

        model.train()

    all_detections = [[] for _ in range(world_size())]
    all_gather_object(all_detections, detections)
    print_once(f"Inferring results for the eval dataset took {time.time() - start_time} seconds")

    if is_root_process():
        all_detections = list(itertools.chain.from_iterable(all_detections))
        ground_truth = COCO(EVAL_ANNOTATION_FILE)
        detections = ground_truth.loadRes(all_detections)
        eval = COCOeval(cocoGt=ground_truth, cocoDt=detections, iouType="bbox")

        eval.evaluate()
        eval.accumulate()

        print(f"Evaluation results at step {step}:")
        eval.summarize()
