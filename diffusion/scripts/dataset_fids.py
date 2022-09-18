import os
from argparse import ArgumentParser
from typing import Iterable
from typing import Tuple

import numpy as np
import torch.distributed
import tqdm
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.distributed import all_gather

from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size
from common.wandb import WANDB_DIR
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.unconditional.consts import DATA_DIR
from diffusion.unconditional.consts import TRAIN_SPLIT
from diffusion.unconditional.data import create_eval_data_loader


def compute_statistics(activations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
	return torch.mean(activations, dim=0).cpu().numpy(), torch.cov(activations.t()).cpu().numpy()


def compute_fid_score(activations1: torch.Tensor, activations2: torch.Tensor) -> float:
	mu1, sigma1 = compute_statistics(activations1)
	mu2, sigma2 = compute_statistics(activations2)
	try:
		return calculate_frechet_distance(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
	except ValueError:
		return -1.0


def compute_activations(data_loader: Iterable[torch.Tensor]) -> torch.Tensor:
	inception_v3 = InceptionV3().cuda()
	inception_v3.eval()

	per_rank_activations = []
	with torch.inference_mode():
		for images in tqdm.tqdm(data_loader, disable=not is_root_process()):
			per_rank_activations.extend(inception_v3(images.cuda()))

	per_rank_activations = torch.flatten(torch.cat(per_rank_activations, dim=0), start_dim=1)
	all_activations = [torch.zeros_like(per_rank_activations) for _ in range(world_size())]
	all_gather(all_activations, per_rank_activations)
	return torch.cat(all_activations, dim=0)


def main():
	torch.distributed.init_process_group("nccl")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	wandb_init(project="fid_analysis", dir=WANDB_DIR)

	parser = ArgumentParser()
	parser.add_argument("--data_dir", type=str, default=os.path.join(DATA_DIR, TRAIN_SPLIT), help="Dataset dir")
	parser.add_argument("--resolution", type=int, default=64, help="Image resolution")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
	parser.add_argument("--dataset_splits", type=str, default="1000,2000", help="Dataset splits")
	args = parser.parse_args()

	dataset_splits = list(map(int, args.dataset_splits.split(",")))
	max_split = max(dataset_splits)

	data_loader = create_eval_data_loader(
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		resolution=args.resolution,
		drop_last=True,
	)
	print_once(f"Training dataset has {len(data_loader.dataset)} examples")

	all_activations = compute_activations(data_loader)

	if is_root_process():
		for num_samples in dataset_splits:
			assert 2 * num_samples <= all_activations.size(0)

			wandb_log({
				"num_samples": num_samples,
				"even_fid_score": compute_fid_score(
					all_activations[:num_samples],
					all_activations[num_samples:2*num_samples],
				),
				"uneven_fid_score": compute_fid_score(
					all_activations[:max_split],
					all_activations[max_split:max_split+num_samples],
				),
			})


if __name__ == "__main__":
	main()
