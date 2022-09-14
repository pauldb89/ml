import itertools
import os
from argparse import ArgumentParser

import numpy as np
import torch.distributed
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.datasets import CocoDetection

from common.distributed import is_root_process
from common.distributed import print_once
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor

from common.distributed import world_size
from common.consts.coco_consts import EVAL_ANNOTATION_FILE
from common.consts.coco_consts import EVAL_ROOT_DIR
from common.consts.coco_consts import FALLBACK_IMAGE_ID


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--scale", type=int, default=600, help="Minimum scale for input image dimension")
    parser.add_argument(
        "--model_config",
        type=str,
        default="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
        help="Detectron2 model config",
    )
    args = parser.parse_args()

    dataset = CocoDetection(
        root=EVAL_ROOT_DIR,
        annFile=EVAL_ANNOTATION_FILE,
    )
    print_once(f"Eval dataset has {len(dataset)} examples")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=lambda x: x,
        sampler=DistributedSampler(dataset=dataset, num_replicas=world_size(), rank=local_rank),
    )

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_config))
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_config)
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("coco_2017_train")
    category_id_mapping = {
        contiguous_id: dataset_id for dataset_id, contiguous_id in metadata.thing_dataset_id_to_contiguous_id.items()
    }

    predictions = []
    for i, batch in enumerate(data_loader):
        for image, gt_annotations in batch:
            image_id = gt_annotations[0]["image_id"] if len(gt_annotations) else FALLBACK_IMAGE_ID
            image = np.asarray(image)
            ret = predictor(image)["instances"]

            for bbox, score, category_id in zip(ret.pred_boxes, ret.scores, ret.pred_classes):
                x1, y1, x2, y2 = bbox.cpu().numpy().tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id_mapping[category_id.item()],
                    "score": score.item(),
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "id": len(predictions),
                })

    all_predictions = [[] for _ in range(world_size())]
    torch.distributed.all_gather_object(all_predictions, predictions)

    if is_root_process():
        predictions = list(itertools.chain.from_iterable(all_predictions))
        ground_truth = COCO(EVAL_ANNOTATION_FILE)
        detections = ground_truth.loadRes(predictions)
        eval = COCOeval(cocoGt=ground_truth, cocoDt=detections, iouType="bbox")

        eval.evaluate()
        eval.accumulate()
        eval.summarize()


if __name__ == "__main__":
    main()
