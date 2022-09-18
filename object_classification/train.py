import datetime
import functools
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim
from torch import distributed
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch.optim.lr_scheduler import StepLR

from common.consts import WANDB_DIR
from common.distributed import print_once
from common.distributed import world_size
from common.solver import Solver
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from object_classification.data import DATASETS
from object_classification.data import get_eval_data_loader
from object_classification.data import get_train_data_loader
from object_classification.evaluate import evaluate
from object_classification.models.inception import InceptionV3
from object_classification.models.model_config import MODEL_CONFIGS
from object_classification.models.model_wrapper import ModelAverageWrapper
from object_classification.models.resnet import ResNet
from object_classification.models.vgg import VGG
from object_classification.models.vgg import VGG_16_CONFIG
from object_classification.models.vision_transformer import VisionTransformer


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    torch.cuda.set_device(local_rank)

    wandb_init(project="object-classification", dir=WANDB_DIR)

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--model", type=str, default="vgg16bn", choices=MODEL_CONFIGS.keys(), help="Model type")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of dataloader workers")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank of -1 for CPU")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=90, help="Number of training batches")
    parser.add_argument("--max_steps", type=int, default=None, help="Number of training batches")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], help="Optimizer name")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD initial momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="SGD weight decay")
    parser.add_argument("--lr_scheduler", type=str, choices=["step", "cosine"], default="step", help="LR scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--warmup_lr_start", type=float, default=1e-6, help="Initial value for learning rate warmup")
    parser.add_argument("--step_lr_epochs", type=int, default=30, help="Epoch frequency to decrease learning rate")
    parser.add_argument("--label_smoothing", type=float, default=0, help="Label smoothing")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Clip gradients to not exceed this L2 norm")
    parser.add_argument("--model_avg_decay", type=float, default=None, help="Exponential model average decay parameter")
    parser.add_argument("--model_avg_steps", type=int, default=32, help="Frequency for updating averaged model")
    args = parser.parse_args()

    wandb_config_update(args)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset_config = DATASETS[args.dataset_name]
    model_config = MODEL_CONFIGS[args.model]

    if args.model == "vgg16":
        model = VGG(layer_group_configs=VGG_16_CONFIG, batch_norm=False, num_classes=dataset_config.num_classes)
    elif args.model == "vgg16bn":
        model = VGG(layer_group_configs=VGG_16_CONFIG, batch_norm=True, num_classes=dataset_config.num_classes)
    elif args.model == "inception_v3":
        model = InceptionV3(num_classes=dataset_config.num_classes)
    elif args.model == "resnet50":
        model = ResNet(num_classes=dataset_config.num_classes)
    elif args.model == "vit16b":
        model = VisionTransformer(
            image_size=model_config.train_resize_dim,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=dataset_config.num_classes,
            label_smoothing=args.label_smoothing,
        )
    else:
        raise ValueError(f"Invalid option {args.model}")

    train_data_loader = get_train_data_loader(
        dataset_name=dataset_config.name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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

    num_steps_per_epoch = len(train_data_loader.dataset) // (args.batch_size * world_size())

    model.to(torch.device("cuda"))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    print(f"Model created on rank: {local_rank}")

    def param_update(avg_param: torch.Tensor, param: torch.Tensor, step: int) -> torch.Tensor:
        if step <= (num_steps_per_epoch * args.warmup_epochs) / args.model_avg_steps:
            return param
        return args.model_avg_decay * avg_param + (1 - args.model_avg_decay) * param
    avg_model = ModelAverageWrapper(model.module, device=torch.device("cuda"), avg_fn=param_update, use_buffers=True)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    warmup_lr_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_lr_start,
        total_iters=args.warmup_epochs * num_steps_per_epoch,
    )
    if args.lr_scheduler == "step":
        main_lr_scheduler = StepLR(optimizer, gamma=0.1, step_size=args.step_lr_epochs * num_steps_per_epoch)
    elif args.lr_scheduler == "cosine":
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warmup_epochs) * num_steps_per_epoch)
    else:
        raise ValueError(f"Unknown learning rate scheduler {args.lr_scheduler}")

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[args.warmup_epochs * num_steps_per_epoch],
    )

    eval_fn = functools.partial(evaluate, data_loader=eval_data_loader)

    solver = Solver(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        max_steps=args.max_steps,
        epochs=args.epochs,
        eval_fn=functools.partial(eval_fn, eval_swag=False),
        evaluate_every_n_steps=num_steps_per_epoch,
        evaluate_avg_model_every_n_steps=num_steps_per_epoch,
        evaluate_at_start=False,
        log_every_n_steps=100,
        max_grad_norm=args.max_grad_norm,
        avg_model=avg_model,
        avg_model_steps=args.model_avg_steps,
        eval_avg_model_fn=functools.partial(eval_fn, eval_swag=True),
    )

    solver.execute()


if __name__ == "__main__":
    main()

