import wandb

from common.distributed import is_root_process


def wandb_init(*args, **kwargs) -> None:
	if is_root_process():
		wandb.init(*args, **kwargs)


def wandb_log(*args, **kwargs) -> None:
	if wandb.run is not None:
		wandb.log(*args, **kwargs)


def wandb_config_update(*args, **kwargs) -> None:
	if wandb.run is not None:
		wandb.config.update(*args, **kwargs)
