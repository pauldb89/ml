import functools
import logging
import os
import time
from argparse import ArgumentParser
from typing import Dict
from typing import Tuple

import torch.distributed
import wandb
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torchvision.transforms import transforms
from transformers import AdamW

from common.distributed import print_once
from common.samplers import set_seeds
from common.solver import Solver
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.common.noise_schedule import LinearNoiseSchedule
from diffusion.conditional.data import Batch
from diffusion.conditional.data import create_eval_data_loader
from diffusion.conditional.data import create_train_data_loader
from diffusion.conditional.evaluate import evaluate
from diffusion.conditional.model import AverageModelWrapper
from diffusion.conditional.model import ConditionalDiffusionModel

log = logging.getLogger("transformers")
log.setLevel(logging.ERROR)


def summarize(
	step: int,
	epoch: int,
	summary_metrics: Dict[str, torch.Tensor],
	models: Dict[str, nn.Module],
	guidance_batch: Batch,
	guidance_strengths: Tuple[float, ...] = (0.5, ),
	# guidance_strengths: Tuple[float, ...] = (-1.0, -0.5, 0., 0.5, 1.0),
	sample_every_n_steps: int = 5_000,
	target_resolution: int = 256,
):
	wandb_log({k: v.item() for k, v in summary_metrics.items()}, step=step)
	if step == 0 or step % sample_every_n_steps != 0:
		return

	image_transform = transforms.ToPILImage()
	with torch.inference_mode():
		for model_name, model in models.items():
			model.eval()
			for guidance_strength in guidance_strengths:
				print(f"Sampling a batch of {len(guidance_batch)} images from model {model_name}...")
				start_time = time.time()
				images = model.sample(batch=guidance_batch, guidance_strength=guidance_strength)
				images = F.interpolate(
					images,
					scale_factor=max(1.0, target_resolution / model.resolution),
					mode="nearest",
				)
				for idx, image in enumerate(torch.chunk(images, chunks=len(guidance_batch), dim=0)):
					pil_image = image_transform(torch.squeeze(image, dim=0))
					wandb_image = wandb.Image(pil_image, caption=guidance_batch.raw_captions[idx])
					wandb_log({f"summary_{model_name}_sample_{idx}": wandb_image}, step=step)
				print(f"Sampling took {time.time() - start_time} seconds...")

			model.train()


def main():
	torch.distributed.init_process_group("nccl")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	set_seeds(local_rank)

	wandb_init(project="conditional-diffusion", dir=WANDB_DIR)

	parser = ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
	parser.add_argument("--resolution", type=int, default=128, help="Image resolution")
	parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps")
	parser.add_argument("--num_diffusion_steps", type=int, default=1_000, help="Number of diffusion steps")
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
	parser.add_argument(
		"--text_encoder",
		type=str,
		choices=[
			"openai/clip-vit-base-patch32",
			"google/t5-v1_1-base",
		],
		default="google/t5-v1_1-base",
		help="Frozen text encoder",
	)
	parser.add_argument(
		"--conditional_dropout",
		type=float,
		default=0.1,
		help="Dropout text embedding for classifier free guidance",
	)
	parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
	parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
	parser.add_argument("--warmup_steps", type=int, default=5_000, help="Number of warmup steps")
	parser.add_argument("--output_dir", type=str, help="Output directory for model checkpoints and artifacts")
	# TODO(pauldb): Maybe try high EMA decay weight.
	parser.add_argument("--ema_decay", type=float, default=0.95, help="Weight decay for exponential model averaging")
	args = parser.parse_args()

	wandb_config_update(args)

	os.makedirs(args.output_dir, exist_ok=True)

	train_data_loader = create_train_data_loader(
		batch_size=args.batch_size,
		max_steps=args.max_steps,
		resolution=args.resolution,
		text_encoder=args.text_encoder,
	)
	print_once(f"Training dataset has {len(train_data_loader.dataset)} examples")

	eval_data_loader = create_eval_data_loader(
		batch_size=args.batch_size,
		resolution=args.resolution,
		text_encoder=args.text_encoder,
	)
	print_once(f"Evaluation dataset has {len(eval_data_loader.dataset)} examples")

	model = ConditionalDiffusionModel(
		text_encoder=args.text_encoder,
		resolution=args.resolution,
		num_diffusion_steps=args.num_diffusion_steps,
		noise_schedule=LinearNoiseSchedule(
			num_diffusion_steps=args.num_diffusion_steps,
			beta_start=args.beta_start,
			beta_end=args.beta_end,
		),
		conditional_dropout=args.conditional_dropout,
	)
	model.cuda()

	model = DistributedDataParallel(model, find_unused_parameters=True)

	def param_update(avg_parameter: torch.Tensor, new_parameter: torch.Tensor, step: int) -> torch.Tensor:
		return args.ema_decay * avg_parameter + (1 - args.ema_decay) * new_parameter

	avg_model = AverageModelWrapper(model.module, device=torch.device("cuda"), avg_fn=param_update, use_buffers=True)

	optimizer = AdamW(
		[param for name, param in model.named_parameters() if not name.startswith("text_encoder")],
		lr=args.lr,
	)

	warmup_lr_scheduler = LinearLR(optimizer, start_factor=1 / args.warmup_steps, total_iters=args.warmup_steps)
	# TODO(pauldb): Maybe try cosine LR schedule.
	main_lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=args.max_steps)
	lr_scheduler = SequentialLR(
		optimizer,
		schedulers=[warmup_lr_scheduler, main_lr_scheduler],
		milestones=[args.warmup_steps],
	)

	eval_fn = functools.partial(evaluate, data_loader=eval_data_loader)

	summary_fn = functools.partial(
		summarize,
		models={"regular": model.module},
		guidance_batch=next(iter(eval_data_loader))[:4],
	)

	solver = Solver(
		model=model,
		optimizer=optimizer,
		lr_scheduler=lr_scheduler,
		train_data_loader=train_data_loader,
		eval_fn=functools.partial(eval_fn, eval_swag=False),
		max_steps=args.max_steps,
		evaluate_every_n_steps=500,
		log_every_n_steps=1_000,
		max_grad_norm=args.max_grad_norm,
		summarize_fn=summary_fn,
		summarize_every_n_steps=1_000,
		avg_model=avg_model,
		eval_avg_model_fn=functools.partial(eval_fn, eval_swag=True),
		evaluate_avg_model_every_n_steps=500,
		snapshot_dir=args.output_dir,
		snapshot_every_n_steps=50_000,
	)

	solver.execute()


if __name__ == "__main__":
	main()
