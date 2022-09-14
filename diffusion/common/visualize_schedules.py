import torch
import wandb

from common.wandb import WANDB_DIR
from common.wandb import wandb_init
from common.wandb import wandb_log
from diffusion.common.noise_schedule import CosineSquaredNoiseSchedule
from diffusion.common.noise_schedule import LinearNoiseSchedule


def main():
	wandb_init(project="diffusion", dir=WANDB_DIR)

	num_diffusion_steps = 1_000
	linear_noise_schedule = LinearNoiseSchedule(
		beta_start=1e-4,
		beta_end=2e-2,
		num_diffusion_steps=num_diffusion_steps,
	)
	cosine_noise_schedule = CosineSquaredNoiseSchedule(num_diffusion_steps=num_diffusion_steps)

	diffusion_steps = torch.arange(num_diffusion_steps)
	linear_alpha_bars = torch.squeeze(linear_noise_schedule.get_alpha_bars(diffusion_steps))
	cosine_alpha_bars = torch.squeeze(cosine_noise_schedule.get_alpha_bars(diffusion_steps))

	for i in range(num_diffusion_steps):
		wandb_log({"linear": linear_alpha_bars[i].item(), "cosine": cosine_alpha_bars[i].item()})


if __name__ == "__main__":
	main()
