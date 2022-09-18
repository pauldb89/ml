import os
from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from common.consts.coco_consts import TRAIN_ROOT_DIR
from common.distributed import is_root_process
from common.distributed import world_size
from common.samplers import set_seeds
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.scripts.dataset_fids import compute_activations
from diffusion.scripts.dataset_fids import compute_fid_score
from diffusion.common.noise_schedule import LinearNoiseSchedule
from diffusion.unconditional.data import create_eval_data_loader
from diffusion.unconditional.model import DiffusionModel


def generate_sample_activations(model: nn.Module, batch_size: int, num_samples: int) -> torch.Tensor:
	with torch.inference_mode():
		assert num_samples % (world_size() * batch_size) == 0
		max_num_samples_per_process = num_samples // world_size()

		images = []
		for _ in tqdm.tqdm(range(0, max_num_samples_per_process, batch_size), disable=not is_root_process()):
			images.append(model.sample(batch_size=batch_size))

	return compute_activations(images)


def main():
	torch.distributed.init_process_group("nccl")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	set_seeds(local_rank)

	wandb_init(project="fid-analysis", dir=WANDB_DIR)

	parser = ArgumentParser()
	parser.add_argument("--data_dir", type=str, default=TRAIN_ROOT_DIR, help="Data directory")
	parser.add_argument("--model_dir", type=str, default="/models/model_fids", help="Model directory")
	parser.add_argument("--model", type=str, required=True, help="")
	parser.add_argument("--attention_block_ids", type=str, default="3", help="Attention blocks")
	parser.add_argument("--resolution", type=int, default=64, help="Resolution")
	parser.add_argument("--num_diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
	parser.add_argument("--sampling_options", type=str, default="1000,2000", help="Sample for which to evaluate FID score")
	parser.add_argument("--num_trials", type=int, default=5, help="Number of trials to get a sense of stability")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
	args = parser.parse_args()

	sampling_options = list(map(int, args.sampling_options.split(",")))
	max_num_samples = max(sampling_options)

	wandb_config_update(args)

	model = DiffusionModel(
		num_diffusion_steps=args.num_diffusion_steps,
		noise_schedule=LinearNoiseSchedule(
			beta_start=1e-4,
			beta_end=2e-2,
			num_diffusion_steps=args.num_diffusion_steps,
		),
		resolution=args.resolution,
		attention_block_ids=frozenset(map(int, args.attention_block_ids.split(",")))
	)
	if is_root_process():
		wandb.use_artifact(args.model).download(root=args.model_dir)
		model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"), map_location="cpu"))

	model.cuda()
	model = DistributedDataParallel(model)
	model.eval()

	sample_activations = generate_sample_activations(
		model=model.module,
		batch_size=args.batch_size,
		num_samples=max_num_samples,
	)

	data_loader = create_eval_data_loader(
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		resolution=args.resolution,
		drop_last=True,
	)
	data_activations = compute_activations(data_loader)

	if is_root_process():
		for num_samples in sampling_options:
			fid_scores = []
			for _ in range(args.num_trials):
				sample_activations = sample_activations[torch.randperm(sample_activations.size(0))]
				fid_scores.append(compute_fid_score(data_activations, sample_activations[:num_samples]))

			wandb_log({
				"num_samples": num_samples,
				"min_fid_score": np.min(fid_scores),
				"max_fid_score": np.max(fid_scores),
				"avg_fid_score": np.mean(fid_scores),
			})


if __name__ == "__main__":
	main()
