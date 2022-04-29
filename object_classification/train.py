import datetime
import functools
import os
from argparse import ArgumentParser
from typing import Callable
from typing import List
from typing import Union

import torch
import torch.optim
from torch import distributed
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler

from object_classification.data import DATASETS
from object_classification.data import get_eval_data_loader
from object_classification.data import get_train_data_loader
from object_classification.distributed_utils import print_once
from object_classification.distributed_utils import world_size
from object_classification.evaluate import evaluate
from object_classification.models.inception import InceptionV3
from object_classification.models.model_config import MODEL_CONFIGS
from object_classification.models.resnet import ResNet
from object_classification.solver import Solver
from object_classification.models.vgg import VGG
from object_classification.models.vgg import VGG_16_CONFIG


LRFunction = Callable[[int], float]


def make_exp_decay_fn(half_life_steps: int) -> LRFunction:
    def get_lr(step: int) -> float:
        return 2 ** (-step / half_life_steps)
    return get_lr


def get_periodic_decay(period_steps: int, decay: float) -> LRFunction:
    def get_lr(step: int) -> float:
        return decay ** (step // period_steps)

    return get_lr


def make_linear_warmup_fn(warmup_steps: int, fn: LRFunction) -> LRFunction:
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps

        return fn(step - warmup_steps)

    return get_lr


class MyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        target_lrs: List[Union[float, List[float]]],
        target_steps: List[int],
        last_epoch: int = -1,
        verbose: bool = False
    ):
        num_param_groups = len(optimizer.param_groups)

        assert len(target_lrs) > 0
        assert len(target_lrs) == len(target_steps)
        for i in range(len(target_lrs)):
            if not isinstance(target_lrs[i], list):
                target_lrs[i] = [target_lrs[i]] * num_param_groups
            else:
                assert len(target_lrs[i]) == num_param_groups

        self.target_lrs = target_lrs
        self.target_steps = target_steps

        super(MyLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.target_steps[0]:
            return [
                base_lr + (target_lr - base_lr) * self.last_epoch / self.target_steps[0]
                for base_lr, target_lr in zip(self.base_lrs, self.target_lrs[0])
            ]

        index = 0
        for i, target_step in enumerate(self.target_steps):
            if self.last_epoch >= target_step:
                index = i

        return self.target_lrs[index]


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    torch.cuda.set_device(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--model", type=str, default="vgg16bn", choices=MODEL_CONFIGS.keys(), help="Model type")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of dataloader workers")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank of -1 for CPU")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=1_000_000, help="Number of training batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD initial momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="SGD weight decay")
    parser.add_argument("--step_lr_epochs", type=int, default=20, help="Epoch frequency to decrease learning rate")
    args = parser.parse_args()

    dataset_config = DATASETS[args.dataset_name]
    model_config = MODEL_CONFIGS[args.model]

    train_data_loader = get_train_data_loader(
        dataset_name=dataset_config.name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        resize_dim=model_config.train_resize_dim
    )
    print_once(f"Training dataset has {len(train_data_loader.dataset)} examples")
    eval_data_loader = get_eval_data_loader(
        dataset_name=dataset_config.name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_dim=model_config.eval_resize_dim,
        crop_dim=model_config.eval_crop_dim,
    )
    print_once(f"Eval dataset has {len(eval_data_loader.dataset) * world_size()} examples")

    if args.model == "vgg16":
        model = VGG(layer_group_configs=VGG_16_CONFIG, batch_norm=False, num_classes=dataset_config.num_classes)
    elif args.model == "vgg16bn":
        model = VGG(layer_group_configs=VGG_16_CONFIG, batch_norm=True, num_classes=dataset_config.num_classes)
    elif args.model == "inception_v3":
        model = InceptionV3(num_classes=dataset_config.num_classes)
    elif args.model == "resnet50":
        model = ResNet(num_classes=dataset_config.num_classes)
    else:
        raise ValueError(f"Invalid option {args.model}")

    model.to(torch.device("cuda"))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    print(f"Model created on rank: {local_rank}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_steps_per_epoch = len(train_data_loader.dataset) // (args.batch_size * world_size())

    # lr_scheduler = StepLR(
    #     optimizer=optimizer,
    #     step_size=args.step_lr_epochs * num_steps_per_epoch,
    #     gamma=0.1,
    # )

    batch_factor = args.batch_size / 32
    lr_scheduler = MyLRScheduler(
        optimizer=optimizer,
        target_lrs=[
            batch_factor * args.lr,
            batch_factor * args.lr * 0.1,
            batch_factor * args.lr * 0.01,
            batch_factor * args.lr * 0.001,
        ],
        target_steps=[
            5 * num_steps_per_epoch,
            30 * num_steps_per_epoch,
            60 * num_steps_per_epoch,
            80 * num_steps_per_epoch,
        ]
    )

    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        max_steps=args.max_steps,
        eval_fn=functools.partial(evaluate, data_loader=eval_data_loader),
        evaluate_every_n_steps=num_steps_per_epoch,
        evaluate_at_start=False,
        log_every_n_steps=100,
    )

    solver.execute()


if __name__ == "__main__":
    main()

