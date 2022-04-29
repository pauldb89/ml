import torch.distributed


def is_root_process() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def print_once(message: str) -> None:
    if is_root_process():
        print(message)


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
