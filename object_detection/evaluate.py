import itertools

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader

from common.distributed import is_root_process
from common.distributed import world_size
from object_detection.coco_consts import EVAL_ANNOTATION_FILE
from object_detection.coco_consts import FALLBACK_IMAGE_ID
from object_detection.detector import Detector


def evaluate(step: int, model: Detector, data_loader: DataLoader) -> None:
    detections = []
    for batch in data_loader:
        preds = model.eval_forward(batch.images)
        for image_classes, image_bboxes, image_labels in zip(preds["classes"], preds["bboxes"], batch.labels):
            image_id = batch.labels[0]["image_id"] if batch.labels else FALLBACK_IMAGE_ID
            for predicted_class, bbox in zip(image_classes, image_bboxes):
                detections.append({
                    "category_id": predicted_class,
                    "bbox": bbox,
                    "image_id": image_id,
                    "id": len(detections),
                    # TODO(pauldb): Fill in the score.
                    "score": 0.5,
                })

    all_detections = [[] for _ in range(world_size())]
    all_gather_object(all_detections, detections)

    if is_root_process():
        all_detections = list(itertools.chain.from_iterable(all_detections))
        ground_truth = COCO(EVAL_ANNOTATION_FILE)
        detections = ground_truth.loadRes(all_detections)
        eval = COCOeval(cocoGt=ground_truth, cocoDt=detections, iouType="bbox")

        eval.evaluate()
        eval.accumulate()

        print(f"Evaluation results at step {step}:")
        eval.summarize()

