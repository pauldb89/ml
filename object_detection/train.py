import functools
import os
from argparse import ArgumentParser

import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ConstantLR

from common.solver import Solver
from object_detection.coco_consts import EVAL_ANNOTATION_FILE
from object_detection.coco_consts import EVAL_ROOT_DIR
from object_detection.coco_consts import TRAIN_ANNOTATION_FILE
from object_detection.coco_consts import TRAIN_ROOT_DIR
from object_detection.data import create_data_loader
from object_detection.detector import Detector
from common.distributed import print_once
from object_detection.evaluate import evaluate

def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--num_images_per_batch", type=int, default=1, help="Number of images per batch")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    train_data_loader = create_data_loader(
        # root_dir=TRAIN_ROOT_DIR,
        # annotation_file=TRAIN_ANNOTATION_FILE,
        root_dir=EVAL_ROOT_DIR,
        annotation_file=EVAL_ANNOTATION_FILE,
        batch_size=args.num_images_per_batch,
        is_train=True,
    )
    print_once(f"Training dataset has {len(train_data_loader)} examples")

    eval_data_loader = create_data_loader(
        root_dir=EVAL_ROOT_DIR,
        annotation_file=EVAL_ANNOTATION_FILE,
        batch_size=args.num_images_per_batch,
        is_train=False,
    )
    print_once(f"Eval dataset has {len(eval_data_loader)} examples")

    model = Detector(num_images_per_batch=args.num_images_per_batch)
    model.cuda()
    model = DistributedDataParallel(model, find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = ConstantLR(optimizer=optimizer, factor=1.0)

    # torch.autograd.set_detect_anomaly(True)
    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        epochs=args.epochs,
        eval_fn=functools.partial(evaluate, data_loader=eval_data_loader),
        evaluate_every_n_steps=1000,
        evaluate_at_start=False,
        log_every_n_steps=100,
    )

    solver.execute()


if __name__ == "__main__":
    main()
