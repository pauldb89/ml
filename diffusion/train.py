import functools
import os
import time
from argparse import ArgumentParser
from typing import Any
from typing import Dict

import torch.distributed
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from common.consts import WANDB_DIR
from common.distributed import is_root_process
from common.distributed import print_once
from common.fid_evaluator import FIDEvaluator
from common.samplers import set_seeds
from common.solver import Solver
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.data import create_eval_data_loader
from diffusion.data import create_train_data_loader
from diffusion.evaluate import evaluate
from diffusion.model import AverageModelWrapper
from diffusion.model import CosineSquaredNoiseSchedule
from diffusion.model import DiffusionModel
from diffusion.model import LinearNoiseSchedule


def summarize(step: int, epoch: int, raw_metrics: Dict[str, Any], models: Dict[str, nn.Module]) -> None:
	wandb_log({"train_loss": raw_metrics["loss"].item()}, step=step)

	if is_root_process():
		with torch.inference_mode():
			for model_name, model in models.items():
				print_once(f"Sampling from model {model_name}...")
				model.eval()
				start_time = time.time()
				images = model.sample(batch_size=9)
				image_grid = make_grid(images, nrow=3)

				image_transform = transforms.ToPILImage()
				image_grid = image_transform(image_grid)

				wandb_log({f"train_samples_{model_name}": wandb.Image(image_grid)}, step=step)
				print_once(f"Generating image samples took {time.time() - start_time} seconds")
				model.train()

def main():
	torch.distributed.init_process_group("nccl")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	set_seeds(local_rank)

	wandb_init(project="diffusion", dir=WANDB_DIR)

	parser = ArgumentParser()
	parser.add_argument("--resolution", type=int, default=128, help="Image resolution")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
	parser.add_argument("--max_steps", type=int, default=1_000_000, help="Number of training iterations")
	parser.add_argument("--warmup_steps", type=int, default=5_000, help="Number of warmup steps")
	parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
	parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Norm for gradient clipping")
	parser.add_argument("--num_diffusion_steps", type=int, default=1_000, help="Number of diffusion steps")
	parser.add_argument(
		"--noise_schedule",
		type=str,
		choices=["linear", "cosine"],
		default="linear",
		help="Noise schedule for forward process",
	)
	parser.add_argument(
		"--beta_start",
		type=float,
		default=1e-4,
		help="Initial beta for when a linear schedule is used for the forward process noise variance",
	)
	parser.add_argument(
		"--beta_end",
		type=float,
		default=2e-2,
		help="Final beta for when a linear schedule is used for the forward process noise variance",
	)
	parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
	parser.add_argument("--ema_decay", type=float, default=0.995, help="Model averaging exponential weight decay")
	parser.add_argument("--num_fid_samples", type=int, default=2048, help="Number of samples for evaluating FID score")

	args = parser.parse_args()
	wandb_config_update(args)

	os.makedirs(args.output_dir, exist_ok=True)

	train_data_loader = create_train_data_loader(
		batch_size=args.batch_size,
		max_steps=args.max_steps,
		resolution=args.resolution,
	)
	print_once(f"Training dataset has {len(train_data_loader.dataset)} examples")
	eval_data_loader = create_eval_data_loader(
		batch_size=args.batch_size,
		resolution=args.resolution,
	)
	print_once(f"Evaluation dataset has {len(eval_data_loader.dataset)} examples")

	fid_evaluator = FIDEvaluator(train_data_loader)

	if args.noise_schedule == "linear":
		noise_schedule = LinearNoiseSchedule(
			beta_start=args.beta_start,
			beta_end=args.beta_end,
			num_diffusion_steps=args.num_diffusion_steps,
		)
	else:
		noise_schedule = CosineSquaredNoiseSchedule(num_diffusion_steps=args.num_diffusion_steps)

	model = DiffusionModel(
		num_diffusion_steps=args.num_diffusion_steps,
		noise_schedule=noise_schedule,
		resolution=args.resolution,
	)
	model.cuda()
	model = DistributedDataParallel(model)

	def param_update(avg_parameter: torch.Tensor, model_parameter: torch.Tensor, steps: int) -> torch.Tensor:
		return args.ema_decay * avg_parameter + (1 - args.ema_decay) * model_parameter
	avg_model = AverageModelWrapper(model.module, device=torch.device("cuda"), avg_fn=param_update, use_buffers=True)

	optimizer = AdamW(model.parameters(), lr=args.lr)
	warmup_lr_scheduler = LinearLR(optimizer, start_factor=1 / args.warmup_steps, total_iters=args.warmup_steps)
	main_lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=args.max_steps)
	lr_scheduler = SequentialLR(
		optimizer,
		schedulers=[warmup_lr_scheduler, main_lr_scheduler],
		milestones=[args.warmup_steps],
	)

	eval_fn = functools.partial(
		evaluate,
		data_loader=eval_data_loader,
		output_dir=args.output_dir,
		sample_batch_size=args.batch_size,
		fid_evaluator=fid_evaluator,
		num_fid_samples=args.num_fid_samples,
	)

	solver = Solver(
		model=model,
		optimizer=optimizer,
		lr_scheduler=lr_scheduler,
		train_data_loader=train_data_loader,
		eval_fn=functools.partial(eval_fn, eval_swag=False),
		evaluate_every_n_steps=50_000,
		max_steps=args.max_steps,
		max_grad_norm=args.max_grad_norm,
		log_every_n_steps=1_000,
		summarize_fn=functools.partial(summarize, models={"regular": model.module, "avg": avg_model}),
		summarize_every_n_steps=5_000,
		avg_model=avg_model,
		eval_avg_model_fn=functools.partial(eval_fn, eval_swag=True),
		evaluate_avg_model_every_n_steps=500_000,
		snapshot_dir=args.output_dir,
		snapshot_every_n_steps=5_000,
	)

	solver.execute()


if __name__ == "__main__":
	main()
