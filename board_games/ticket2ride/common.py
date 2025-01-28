import contextlib
import time
from typing import Generator

import wandb


@contextlib.contextmanager
def timer(key: str, step: int) -> Generator[None, None, None]:
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    wandb.log({key: elapsed_time}, step=step)
