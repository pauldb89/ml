import collections
import time
from typing import Optional

import torch
from torch import nn
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader

from common.distributed import get_rank
from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size


def evaluate(step: int, model: nn.Module, data_loader: DataLoader, log_progress_every_n_steps: Optional[int] = 10):
    start_time = time.time()
    print_once(f"Running inference on eval dataset at step {step}")

    model.eval()

    num_examples = 0
    num_batches = 0
    losses = collections.defaultdict(float)
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, start=1):
            batch_losses = model(batch)
            for k, v in batch_losses.items():
                losses[k] += v.item()

            num_examples += len(batch)
            num_batches += 1
            if log_progress_every_n_steps is not None and batch_idx % log_progress_every_n_steps == 0:
                print_once(f"{num_examples} examples processed on rank {get_rank()}...")

    losses = {k: v / num_batches for k, v in losses.items()}

    model.train()

    all_losses = [{} for _ in range(world_size())]
    all_gather_object(all_losses, losses)
    print_once(f"Inferring results for the eval dataset took {time.time() - start_time} seconds")

    if is_root_process():
        agg_losses = collections.defaultdict(float)
        for loss in all_losses:
            for k, v in loss.items():
                agg_losses[k] += v

        metrics = "\t".join([f"{k}: {v / world_size()}" for k, v in agg_losses.items()])
        print(f"Evaluation results at {step}: {metrics}")
