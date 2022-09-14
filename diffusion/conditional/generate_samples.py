import time
from argparse import ArgumentParser

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms import transforms

from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.common.noise_schedule import LinearNoiseSchedule
from diffusion.conditional.data import create_eval_data_loader
from diffusion.conditional.model import AverageModelWrapper
from diffusion.conditional.model import ConditionalDiffusionModel


def main():
	torch.distributed.init_process_group("nccl")
	torch.cuda.set_device(0)

	wandb_init(project="conditional-diffusion", dir=WANDB_DIR)

	parser = ArgumentParser()
	parser.add_argument("--model", type=str, help="Model file")
	parser.add_argument("--guidance_strength", type=float, default=-1.0, help="Classifier free guidance strength")
	parser.add_argument("--batch_size", type=int, default=4, help="Number of images to sample")
	parser.add_argument("--avg_model", type=int, default=0, choices=[0, 1], help="Whether this is an averaged model")
	args = parser.parse_args()

	wandb_config_update(args)

	model = ConditionalDiffusionModel(
		text_encoder="google/t5-v1_1-base",
		resolution=128,
		num_diffusion_steps=1_000,
		noise_schedule=LinearNoiseSchedule(
			num_diffusion_steps=1_000,
			beta_start=1e-4,
			beta_end=2e-2,
		),
		conditional_dropout=1.0,
	)

	model.cuda()

	if args.avg_model:
		model = DistributedDataParallel(model)
		model = AverageModelWrapper(model, device=torch.device("cuda"), avg_fn=lambda x, y, z: x, use_buffers=True)

	model.load_state_dict(torch.load(args.model, map_location="cuda"))

	if args.avg_model:
		model = model.module.module

	eval_data_loader = create_eval_data_loader(
		batch_size=args.batch_size,
		resolution=128,
		text_encoder="google/t5-v1_1-base",
	)
	print(f"Eval dataset has {len(eval_data_loader.dataset)} examples")

	batch = next(iter(eval_data_loader))
	image_transform = transforms.ToPILImage()

	print(f"Sampling a batch of {len(batch)} images from model...")
	start_time = time.time()
	with torch.inference_mode():
		model.eval()
		images = model.sample(batch=batch, guidance_strength=args.guidance_strength)
		model.train()

	for idx, image in enumerate(torch.chunk(images, chunks=len(batch), dim=0)):
		pil_image = image_transform(torch.squeeze(image, dim=0))
		wandb_image = wandb.Image(pil_image, caption=batch.raw_captions[idx])
		wandb_log({f"summary_sample_{idx}": wandb_image})
	print(f"Sampling took {time.time() - start_time} seconds...")


if __name__ == "__main__":
	main()
