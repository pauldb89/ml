import collections
import os
import time

import torch
from torch import nn
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from common.distributed import get_rank
from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size


def evaluate(
    step: str,
    model: nn.Module,
    data_loader: DataLoader,
    output_dir: str,
    log_progress_every_n_steps: int = 10,
    sample_tile_size: int = 4,
    min_temperature: float = 0.6,
    max_temperature: float = 1.0,
    resolution: int = 64,
) -> None:
    print_once(f"Running inference on eval dataset at step {step}")
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        losses = collections.defaultdict(list)
        for batch_idx, batch in enumerate(data_loader, start=1):
            batch_losses = model(batch)

            for key, loss in batch_losses.items():
                if "loss" in key:
                    losses[key].append(loss)

            if batch_idx % log_progress_every_n_steps == 0:
                print_once(f"{batch_idx * batch.size(0)} examples processed on rank {get_rank()}...")

    agg_losses = {k: torch.mean(torch.stack(loss, dim=0)).item() for k, loss in losses.items()}
    all_losses = [{} for _ in range(world_size())]
    all_gather_object(all_losses, agg_losses)
    print_once(f"Running inference on eval dataset took {time.time() - start_time} seconds")

    if is_root_process():
        losses = collections.defaultdict(list)
        for per_rank_losses in all_losses:
            for k, loss in per_rank_losses.items():
                losses[k].append(loss)

        metrics = {k: sum(loss) / len(loss) for k, loss in losses.items()}
        formatted_metrics = "\t".join([f"{k}: {v / world_size()}" for k, v in metrics.items()])
        print(f"Evaluation results at {step}: {formatted_metrics}")

        print("Sampling images...")
        start_time = time.time()
        with torch.no_grad():
            temperature = min_temperature
            image_transform = transforms.ToPILImage()
            while temperature <= max_temperature:
                images = model.sample(batch_size=sample_tile_size * sample_tile_size, temperature=temperature)
                images = images.view(sample_tile_size, sample_tile_size, 3, resolution, resolution)
                images = images.permute(2, 0, 3, 1, 4)
                images = images.reshape(3, resolution * sample_tile_size, resolution * sample_tile_size)
                image_path = os.path.join(output_dir, f"eval_{step}_{temperature:.1f}.png")
                print(f"Saving sampled images to {image_path}")
                image_transform(images).save(image_path)
                temperature += 0.1

        print(f"Sampling images took {time.time() - start_time} seconds")

    model.train()
