import collections
import math
import os
import time
from typing import Iterable
from typing import Iterator
from typing import Optional

import torch
import wandb
from torch import nn
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from common.distributed import get_rank
from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size
from common.fid_evaluator import FIDEvaluator
from common.wandb import wandb_log


class Sampler(Iterable[torch.Tensor]):
	def __init__(self, model: nn.Module, batch_size: int):
		self.model = model
		self.batch_size = batch_size

	def __next__(self) -> torch.Tensor:
		with torch.inference_mode():
			return self.model.sample(batch_size=self.batch_size)

	def __iter__(self) -> Iterator[torch.Tensor]:
		return self


def evaluate(
	step: int,
	model: nn.Module,
	data_loader: DataLoader,
	output_dir: str,
	fid_evaluator: FIDEvaluator,
	num_fid_samples: Optional[int],
	sample_batch_size: int,
	eval_swag: bool,
	log_progress_every_n_steps: int = 10,
) -> None:
	model.eval()
	start_time = time.time()
	formatted_step = step if not eval_swag else f"{step} SWAG"
	print_once(f"Running inference on eval dataset at step {formatted_step}")

	per_rank_metrics = collections.defaultdict(list)
	with torch.inference_mode():
		for batch_idx, batch in enumerate(data_loader, start=1):
			metrics = model(batch)
			for key, metric in metrics.items():
				per_rank_metrics[key].append(metric)

		if batch_idx % log_progress_every_n_steps == 0:
			print_once(f"Finished evaluating {batch_idx * batch.size(0)} examples on rank {get_rank()}")

	per_rank_metrics = {
		key: torch.mean(torch.stack(metrics, dim=0)).item() for key, metrics in per_rank_metrics.items()
	}

	all_metrics = [{} for _ in range(world_size())]
	all_gather_object(all_metrics, per_rank_metrics)

	print_once(f"Running inference on eval dataset took {time.time() - start_time} seconds...")

	agg_metrics = collections.defaultdict(list)
	for per_rank_metrics in all_metrics:
		for key, metric in per_rank_metrics.items():
			agg_metrics[key].append(metric)

	agg_metrics = {key: sum(metrics) / len(metrics) for key, metrics in agg_metrics.items()}

	sampler = Sampler(model=model, batch_size=sample_batch_size)
	fid_score = fid_evaluator.fid_score(sampler, num_samples=num_fid_samples)
	if fid_score is not None:
		agg_metrics["fid_score"] = fid_score

	if is_root_process():
		eval_prefix = "eval" if not eval_swag else "eval_swag"
		wandb_log({f"{eval_prefix}_{key}": value for key, value in agg_metrics.items()}, step=step)
		formatted_metrics = "\t".join([f"{key}: {metric}" for key, metric in agg_metrics.items()])
		print(f"Evaluation results at {formatted_step}: {formatted_metrics}")

		with torch.inference_mode():
			start_time = time.time()
			images = model.sample(batch_size=sample_batch_size)
			image_grid = make_grid(images, nrow=int(math.sqrt(sample_batch_size)))

			image_path = os.path.join(output_dir, f"{eval_prefix}_step_{step}.png")
			image_transform = transforms.ToPILImage()
			image_grid = image_transform(image_grid)

			wandb_log({f"{eval_prefix}_samples": wandb.Image(image_grid)}, step=step)

			print(f"Saving images to {image_path}")
			image_grid.save(image_path)

			print(f"Sampling {sample_batch_size} images took {time.time() - start_time} seconds...")

	model.train()
