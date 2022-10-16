import wandb

from common.distributed import is_root_process

WANDB_DIR = "/wandb"


def wandb_init(*args: object, **kwargs: object) -> None:
	if is_root_process():
		wandb.init(*args, **kwargs)


def wandb_log(*args, **kwargs) -> None:
	if wandb.run is not None:
		wandb.log(*args, **kwargs)


def wandb_config_update(*args, **kwargs) -> None:
	if wandb.run is not None:
		wandb.config.update(*args, **kwargs)
