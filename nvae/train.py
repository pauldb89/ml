import functools
import os
from argparse import ArgumentParser

import torch.cuda
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

from common.distributed import print_once
from common.distributed import world_size
from common.samplers import set_seeds
from common.solver import Solver
from nvae.consts import CELEBA_TRAIN_DIR
from nvae.consts import CELEBA_VAL_DIR
from nvae.data import create_data_loader
from nvae.evaluate import evaluate
from nvae.model import NVAE


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    set_seeds(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    args = parser.parse_args()

    train_data_loader = create_data_loader(root=CELEBA_TRAIN_DIR, batch_size=args.batch_size, is_train=True)
    print_once(f"Training dataset has {len(train_data_loader.dataset)} examples")

    eval_data_loader = create_data_loader(root=CELEBA_VAL_DIR, batch_size=args.batch_size, is_train=False)
    print_once(f"Eval dataset has {len(eval_data_loader.dataset)} examples")

    model = NVAE(num_latent_groups=26, encoder_resolution_group_offsets=frozenset([8, 14, 18, 22]), num_encoder_channels=10)
    print_once("Model created on rank 0")
    model.cuda()
    model = DistributedDataParallel(model, find_unused_parameters=True)
    print_once("Distributed model created on rank 0")

    steps_per_epoch = len(train_data_loader.dataset) // (args.batch_size * world_size())

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)

    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        eval_fn=functools.partial(evaluate, data_loader=eval_data_loader),
        epochs=args.epochs,
        evaluate_every_n_steps=steps_per_epoch,
        summarize_fn=model.module.summarize,
        summarize_every_n_steps=1,
        log_every_n_steps=50,
    )

    print_once("Starting to train")

    solver.execute()


if __name__ == "__main__":
    main()
