import os
from argparse import ArgumentParser

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

from common.consts.coco_consts import TRAIN_ROOT_DIR
from common.distributed import is_root_process
from common.samplers import set_seeds
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.scripts.dataset_fids import compute_activations
from diffusion.scripts.dataset_fids import compute_fid_score
from diffusion.scripts.model_fids import generate_sample_activations
from diffusion.common.noise_schedule import LinearNoiseSchedule
from diffusion.unconditional.data import create_eval_data_loader
from diffusion.unconditional.model import DiffusionModel


def main():
	torch.distributed.init_process_group("nccl")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	set_seeds(local_rank)

	wandb_init(project="fid-analysis", dir=WANDB_DIR)

	parser = ArgumentParser()
	parser.add_argument("--data_dir", type=str, default=TRAIN_ROOT_DIR, help="Data directory")
	parser.add_argument("--model_dir", type=str, default="/models/model_fids", help="Model directory")
	parser.add_argument("--model_versions", type=str, required=True, help="List of model versions")
	parser.add_argument("--attention_block_ids", type=str, default="3", help="Attention blocks")
	parser.add_argument("--resolution", type=int, default=64, help="Resolution")
	parser.add_argument("--num_diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
	parser.add_argument("--num_samples", type=int, default=8192, help="Sample for which to evaluate FID score")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
	args = parser.parse_args()

	wandb_config_update(args)

	data_loader = create_eval_data_loader(
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		resolution=args.resolution,
		drop_last=True,
	)
	data_activations = compute_activations(data_loader)

	for model_version in args.model_versions.split(","):
		base_model = DiffusionModel(
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
			wandb.use_artifact(model_version).download(root=args.model_dir)
			base_model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"), map_location="cpu"))

		base_model.cuda()
		model = DistributedDataParallel(base_model)
		model.eval()

		sample_activations = generate_sample_activations(
			model=model.module,
			batch_size=args.batch_size,
			num_samples=args.num_samples,
		)

		if is_root_process():
			wandb_log({
				"model_version": model_version,
				"fid_scores": compute_fid_score(data_activations, sample_activations),
			})


if __name__ == "__main__":
	main()
