import functools
import os
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from common.distributed import print_once
from common.solver import Solver
from neural_style_transfer.consts import COCO_EVAL
from neural_style_transfer.consts import COCO_TRAIN
from neural_style_transfer.consts import WIKIART_EVAL
from neural_style_transfer.consts import WIKIART_TRAIN
from neural_style_transfer.data import create_eval_data_loader
from neural_style_transfer.data import create_train_data_loader
from neural_style_transfer.evaluate import evaluate
from neural_style_transfer.model import StyleTransfer


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--style_weight", type=float, default=50.0, help="How much weight to assign to style loss")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of data loader workers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=20_000, help="Number of training steps")
    parser.add_argument("--snapshot_dir", type=str, default=None, help="Model snapshot directory")
    parser.add_argument("--snapshot_every_n_steps", type=int, default=1000, help="Snapshotting frequency")
    args = parser.parse_args()

    train_data_loader = create_train_data_loader(
        content_root_dir=COCO_TRAIN,
        style_root_dir=WIKIART_TRAIN,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print_once(f"Training dataset has {len(train_data_loader.dataset)} examples")
    eval_data_loader = create_eval_data_loader(
        content_root_dir=COCO_EVAL,
        style_root_dir=WIKIART_EVAL,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print_once(f"Eval dataset has {len(eval_data_loader.dataset)} examples")

    model = StyleTransfer(
        content_feature_name="layer4",
        style_feature_names=["layer1", "layer2", "layer3", "layer4"],
        style_weight=args.style_weight,
    )
    model.cuda()
    model = DistributedDataParallel(model, find_unused_parameters=True)

    params = []
    param_names = []
    for pn, p in model.named_parameters():
        if "decoder" in pn:
            param_names.append(pn)
            params.append(p)

    print_once(f"Optimizing params {param_names}")
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    warmup_steps = args.max_steps // 10
    warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps - warmup_steps)
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[warmup_steps],
    )

    if args.snapshot_dir is not None:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        eval_fn=functools.partial(evaluate, data_loader=eval_data_loader),
        max_steps=args.max_steps,
        epochs=1000,
        evaluate_every_n_steps=1000,
        summarize_fn=model.module.summarize,
        summarize_every_n_steps=100,
        log_every_n_steps=100,
        snapshot_dir=args.snapshot_dir,
        snapshot_every_n_steps=args.snapshot_every_n_steps,
    )

    solver.execute()


if __name__ == "__main__":
    main()
