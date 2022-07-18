import functools
import os
from argparse import ArgumentParser
from typing import List

import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from common.distributed import print_once
from common.solver import Solver
from object_detection.coco_consts import EVAL_ANNOTATION_FILE
from object_detection.coco_consts import EVAL_ROOT_DIR
from object_detection.coco_consts import TRAIN_ANNOTATION_FILE
from object_detection.coco_consts import TRAIN_ROOT_DIR
from object_detection.data import create_train_data_loader, create_eval_data_loader
from object_detection.detector import Detector
from object_detection.evaluate import evaluate


def lr_schedule(epoch: int, warmup_steps: int, thresholds: List[int]) -> float:
    if epoch < warmup_steps:
        return (epoch + 1) / warmup_steps

    ret = 1
    for threshold in thresholds:
        if epoch >= threshold:
            ret *= 0.1

    return ret


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--num_images_per_batch", type=int, default=1, help="Number of images per batch")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=90_000, help="Number of training iterations")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of learning rate warmup steps")
    args = parser.parse_args()

    train_data_loader = create_train_data_loader(
        root_dir=TRAIN_ROOT_DIR,
        annotation_file=TRAIN_ANNOTATION_FILE,
        batch_size=args.num_images_per_batch,
        max_steps=args.max_steps,
    )
    print_once(f"Training dataset has {len(train_data_loader)} examples")

    eval_data_loader = create_eval_data_loader(
        root_dir=EVAL_ROOT_DIR,
        annotation_file=EVAL_ANNOTATION_FILE,
        batch_size=8,
    )
    print_once(f"Eval dataset has {len(eval_data_loader)} examples")

    model = Detector()
    for pn, p in model.named_parameters():
        if "backbone.resnet.conv1" in pn or "backbone.resnet.layer1" in pn:
            p.requires_grad = False

    model.cuda()
    model = DistributedDataParallel(model, find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=functools.partial(lr_schedule, warmup_steps=args.warmup_steps, thresholds=[60_000, 80_000]),
    )

    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        epochs=1,
        max_steps=args.max_steps,
        eval_fn=functools.partial(evaluate, data_loader=eval_data_loader),
        evaluate_every_n_steps=2500,
        log_every_n_steps=100,
        summarize_fn=model.module.summarize,
        summarize_every_n_steps=100,
    )

    solver.execute()


if __name__ == "__main__":
    main()
