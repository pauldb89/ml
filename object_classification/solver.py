import enum
import time
from collections import deque
from typing import Optional
from typing import Protocol

import numpy as np
import torch.optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torch.utils.data import DataLoader

from object_classification.distributed_utils import is_root_process
from object_classification.distributed_utils import world_size


class TimeStat:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.end_time = None

    def end(self) -> None:
        assert self.start_time is not None
        self.end_time = time.perf_counter()

    def get(self) -> float:
        assert self.start_time is not None
        assert self.end_time is not None
        return self.end_time - self.start_time


class TimeKey(enum.IntEnum):
    DATA_LOAD = 1
    FORWARD = 2
    BACKWARD = 3


class Timer:
    def __init__(self):
        self.time_stats = {
            TimeKey.DATA_LOAD: TimeStat(),
            TimeKey.FORWARD: TimeStat(),
            TimeKey.BACKWARD: TimeStat(),
        }

    def start(self, time_key: TimeKey) -> None:
        self.time_stats[time_key].start()

    def end(self, time_key: TimeKey) -> None:
        self.time_stats[time_key].end()

    def get(self, time_key: TimeKey) -> float:
        return self.time_stats[time_key].get()


class Throughput:
    def __init__(self):
        self.time_stat = TimeStat()
        self.start_value = None
        self.end_value = None

    def start(self, value: int) -> None:
        self.time_stat.start()
        self.start_value = value
        self.end_value = None

    def end(self, value: int) -> None:
        self.time_stat.end()
        assert self.start_value is not None
        self.end_value = value

    def get(self) -> float:
        return (self.end_value - self.start_value) / self.time_stat.get()


class EvalFunction(Protocol):
    def __call__(self, model: DistributedDataParallel, step: int) -> None:
        ...


class Solver:
    def __init__(
        self,
        model: DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: _LRScheduler,
        train_data_loader: DataLoader,
        max_steps: int,
        eval_fn: EvalFunction,
        evaluate_every_n_steps: Optional[int] = None,
        evaluate_at_start: bool = False,
        evaluate_at_end: bool = True,
        log_every_n_steps: Optional[int] = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data_loader = train_data_loader
        self.max_steps = max_steps
        self.eval_fn = eval_fn
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.evaluate_at_start = evaluate_at_start
        self.evaluate_at_end = evaluate_at_end
        self.log_every_n_steps = log_every_n_steps

    def execute(self):
        if self.evaluate_at_start:
            self.eval_fn(model=self.model, step=0)

        self.optimizer.zero_grad()
        scaler = GradScaler()
        loss_history = deque(maxlen=100)

        timer = Timer()
        timer.start(TimeKey.DATA_LOAD)
        throughput = Throughput()
        throughput.start(0)

        for step, batch in enumerate(self.train_data_loader):
            timer.end(TimeKey.DATA_LOAD)

            if step > self.max_steps:
                break

            timer.start(TimeKey.FORWARD)
            with autocast():
                loss = self.model(batch)
                loss_history.append(loss.item())

            timer.end(TimeKey.FORWARD)

            timer.start(TimeKey.BACKWARD)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            timer.end(TimeKey.BACKWARD)

            if is_root_process() and self.log_every_n_steps is not None and step % self.log_every_n_steps == 0:
                throughput.end((step+1) * batch.size() * world_size())

                print(
                    f"Step {step}: LR {self.lr_scheduler.get_last_lr()[0]:.8f}\t Loss {np.mean(loss_history):.5f}\t"
                    f"Data load time {timer.get(TimeKey.DATA_LOAD):.5f} seconds\t"
                    f"Forward time {timer.get(TimeKey.FORWARD):.5f} seconds\t"
                    f"Backward time {timer.get(TimeKey.BACKWARD):.5f} seconds\t"
                    f"Throughput {throughput.get():5f} examples/second"
                )

                throughput.start((step+1) * batch.size() * world_size())

            if self.evaluate_every_n_steps is not None and step > 0 and step % self.evaluate_every_n_steps == 0:
                self.eval_fn(model=self.model, step=step)

            timer.start(TimeKey.DATA_LOAD)

        if self.evaluate_at_end:
            self.eval_fn(model=self.model, step=self.max_steps)
