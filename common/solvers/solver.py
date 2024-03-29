import enum
import os
import time
from collections import deque
from typing import Optional, Callable, Dict, Any
from typing import Protocol

import numpy as np
import torch.optim
import wandb
from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torch.utils.data import DataLoader

from common.distributed import get_rank
from common.distributed import is_root_process
from common.distributed import world_size
from common.wandb import wandb_log


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


class ModelType:
    REGULAR = "regular"
    AVERAGE = "avg"


class EvalFunction(Protocol):
    def __call__(self, step: int, model: nn.Module) -> None:
        ...


class Solver:
    def __init__(
        self,
        model: DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: _LRScheduler,
        train_data_loader: DataLoader,
        eval_fn: EvalFunction,
        eval_avg_model_fn: Optional[EvalFunction] = None,
        max_steps: Optional[int] = None,
        epochs: int = 1,
        evaluate_every_n_steps: Optional[int] = None,
        evaluate_avg_model_every_n_steps: Optional[int] = None,
        evaluate_at_start: bool = False,
        evaluate_at_end: bool = True,
        log_every_n_steps: Optional[int] = 10,
        summarize_fn: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        summarize_every_n_steps: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        avg_model: Optional[nn.Module] = None,
        avg_model_steps: Optional[int] = 32,
        snapshot_dir: Optional[str] = None,
        snapshot_every_n_steps: Optional[int] = None,
    ):
        assert eval_avg_model_fn is None or avg_model is not None, (
            "eval_avg_model_fn can be defined only when avg_model is defined"
        )

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data_loader = train_data_loader
        self.eval_fn = eval_fn
        self.eval_avg_model_fn = eval_avg_model_fn
        self.max_steps = max_steps
        self.epochs = epochs
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.evaluate_avg_model_every_n_steps = evaluate_avg_model_every_n_steps
        self.evaluate_at_start = evaluate_at_start
        self.evaluate_at_end = evaluate_at_end
        self.log_every_n_steps = log_every_n_steps
        self.summarize_fn = summarize_fn
        self.summarize_every_n_steps = summarize_every_n_steps
        self.max_grad_norm = max_grad_norm
        self.avg_model = avg_model
        self.avg_model_steps = avg_model_steps
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_n_steps = snapshot_every_n_steps

    def save_model(self, model: nn.Module, step: int, model_type: str) -> None:
        model_file = os.path.join(self.snapshot_dir, f"step-{step}-{model_type}.pt")
        torch.save(model.state_dict(), model_file)

        if wandb.run is not None:
            artifact = wandb.Artifact(f'{wandb.run.project}-{wandb.run.name}-{model_type}', type="model")
            artifact.add_file(model_file, "model.pt")
            wandb.log_artifact(artifact)

    def execute(self):
        if self.evaluate_at_start:
            self.eval_fn(model=self.model.module, step=0)

        self.optimizer.zero_grad()
        scaler = GradScaler()
        loss_history = deque(maxlen=100)

        timer = Timer()
        timer.start(TimeKey.DATA_LOAD)
        throughput = Throughput()
        throughput.start(0)

        step = 0
        for epoch in range(self.epochs):
            for batch in self.train_data_loader:
                timer.end(TimeKey.DATA_LOAD)

                if self.max_steps is not None and step >= self.max_steps:
                    break

                timer.start(TimeKey.FORWARD)
                with autocast():
                    output = self.model(batch)
                    loss = output["loss"]
                    loss_history.append(loss.item())

                timer.end(TimeKey.FORWARD)

                timer.start(TimeKey.BACKWARD)
                scaler.scale(loss).backward()

                if self.max_grad_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                timer.end(TimeKey.BACKWARD)

                if self.avg_model is not None and self.avg_model_steps is not None and step % self.avg_model_steps == 0:
                    self.avg_model.update_parameters(self.model)

                if (is_root_process() and
                        self.summarize_fn is not None and
                        self.summarize_every_n_steps is not None and
                        step % self.summarize_every_n_steps == 0):
                    self.summarize_fn(step, epoch, output)

                if is_root_process() and self.log_every_n_steps is not None and step % self.log_every_n_steps == 0:
                    throughput.end((step+1) * len(batch) * world_size())

                    last_lr = self.lr_scheduler.get_last_lr()[0]
                    wandb_log(
                        {
                            "lr": last_lr,
                            "epoch": epoch,
                            "avg_loss": np.mean(loss_history),
                            "data_load_time": timer.get(TimeKey.DATA_LOAD),
                            "forward_time": timer.get(TimeKey.FORWARD),
                            "backward_time": timer.get(TimeKey.BACKWARD),
                            "throughput": throughput.get(),
                        },
                        step=step,
                    )
                    print(
                        f"Step {step}: Epoch {epoch}\t"
                        f"LR {last_lr:.8f}\t"
                        f"Loss {np.mean(loss_history):.5f}\t"
                        f"Data load time {timer.get(TimeKey.DATA_LOAD):.5f} seconds\t"
                        f"Forward time {timer.get(TimeKey.FORWARD):.5f} seconds\t"
                        f"Backward time {timer.get(TimeKey.BACKWARD):.5f} seconds\t"
                        f"Throughput {throughput.get():5f} examples/second"
                    )

                    throughput.start((step+1) * len(batch) * world_size())

                if self.evaluate_every_n_steps is not None and step > 0 and step % self.evaluate_every_n_steps == 0:
                    self.eval_fn(model=self.model.module, step=step)
                if (self.avg_model is not None and
                        self.evaluate_avg_model_every_n_steps is not None and
                        step > 0 and step % self.evaluate_avg_model_every_n_steps == 0):
                    self.eval_avg_model_fn(model=self.avg_model, step=step)

                if (is_root_process()
                        and self.snapshot_dir is not None and self.snapshot_every_n_steps is not None
                        and step > 0 and step % self.snapshot_every_n_steps == 0):
                    self.save_model(model=self.model.module, step=step, model_type=ModelType.REGULAR)

                    if self.avg_model is not None:
                        self.save_model(self.avg_model, step=step, model_type=ModelType.AVERAGE)

                timer.start(TimeKey.DATA_LOAD)

                step += 1

        if self.evaluate_at_end:
            self.eval_fn(model=self.model.module, step=step)
            if self.avg_model is not None:
                self.eval_avg_model_fn(model=self.avg_model, step=step)

        if (is_root_process()
                and self.snapshot_dir is not None and self.snapshot_every_n_steps is not None):
            self.save_model(self.model.module, step=step, model_type=ModelType.REGULAR)

            if self.avg_model is not None:
                self.save_model(self.avg_model, step=step, model_type=ModelType.AVERAGE)
