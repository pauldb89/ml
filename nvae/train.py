import functools
import os
from argparse import ArgumentParser

import torch.cuda
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size
from common.samplers import set_seeds
from common.solver import Solver
from nvae.consts import DATASETS_DIR
from nvae.data import create_eval_data_loader
from nvae.data import create_train_data_loader
from nvae.evaluate import evaluate
from nvae.model import LossConfig
from nvae.model import NVAE


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    set_seeds(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num_encoder_channels",
        type=int,
        default=60,
        help="Number of channels for initial processing")
    parser.add_argument("--num_latent_groups", type=int, default=35, help="Number of latent groups")
    parser.add_argument(
        "--resolution_group_offsets",
        type=str,
        default="19,29",
        help="Groups after which resolution is decreased while encoding"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="Weight decay")
    parser.add_argument("--eta_min", type=float, default=1e-4, help="Cosine schedule minimum learning rate")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    # Parameters defining the schedule for the KL divergence weight. For the first kl_const_portion steps (out of
    # the total training steps) the weight of the KL loss is kept constant at kl_min_coeff. It is then increased
    # gradually to 1 over kl_anneal_portion steps. The last part of the training schedule maintains the KL
    # coefficient constant at 1.
    parser.add_argument(
        "--kl_const_portion",
        type=float,
        default=1e-4,
        help="Initial training schedule portion with the KL coeff at kl_min_coeff"
    )
    parser.add_argument(
        "--kl_anneal_portion",
        type=float,
        default=0.3,
        help="Training schedule portion where the KL coeff is gradually increased to 1"
    )
    parser.add_argument("--kl_min_coeff", type=float, default=1e-4, help="Initial KL coefficient value")
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Target weight for regularization losses")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data_loader = create_train_data_loader(root=DATASETS_DIR, batch_size=args.batch_size)
    print_once(f"Training dataset has {len(train_data_loader.dataset)} examples")

    eval_data_loader = create_eval_data_loader(root=DATASETS_DIR, batch_size=args.batch_size)
    print_once(f"Eval dataset has {len(eval_data_loader.dataset)} examples")

    steps_per_epoch = len(train_data_loader.dataset) // (args.batch_size * world_size())
    total_steps = args.epochs * steps_per_epoch

    model = NVAE(
        loss_config=LossConfig(
            kl_const_steps=args.kl_const_portion * total_steps,
            kl_anneal_steps=args.kl_anneal_portion * total_steps,
            kl_min_coeff=args.kl_min_coeff,
            reg_weight=args.reg_weight,
        ),
        num_encoder_channels=args.num_encoder_channels,
        num_latent_groups=args.num_latent_groups,
        encoder_resolution_group_offsets=frozenset(map(int, args.resolution_group_offsets.split(",")))
    )

    print_once("Model created on rank 0")
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DistributedDataParallel(model, find_unused_parameters=True)
    print_once("Distributed model created on rank 0")

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-3)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1 / (args.warmup_epochs * steps_per_epoch),
        total_iters=args.warmup_epochs * steps_per_epoch,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - args.warmup_epochs) * steps_per_epoch,
        eta_min=args.eta_min,
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs * steps_per_epoch],
    )

    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        eval_fn=functools.partial(evaluate, data_loader=eval_data_loader, output_dir=args.output_dir),
        epochs=args.epochs,
        evaluate_every_n_steps=steps_per_epoch,
        summarize_fn=model.module.summarize,
        summarize_every_n_steps=100,
        log_every_n_steps=100,
        snapshot_dir=args.output_dir,
        snapshot_every_n_steps=steps_per_epoch,
    )

    print_once("Starting to train")

    solver.execute()


if __name__ == "__main__":
    main()
