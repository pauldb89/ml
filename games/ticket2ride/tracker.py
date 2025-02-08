import collections
import contextlib
import time
from dataclasses import field, dataclass
from typing import Generator

import numpy as np


@dataclass
class Tracker:
    scopes: list[str] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=lambda: collections.defaultdict(list))

    def log_value(self, metric_name: str, value: float | np.float64) -> None:
        metric_key = "/".join(self.scopes + [metric_name])
        self.metrics[metric_key].append(float(value))

    @contextlib.contextmanager
    def scope(self, name: str) -> Generator[None, None, None]:
        self.scopes.append(name)
        yield
        self.scopes.pop()

    @contextlib.contextmanager
    def timer(self, metric: str) -> Generator[None, None, None]:
        start_time = time.time()
        yield
        self.log_value(metric, time.time() - start_time)

    def report(self) -> dict[str, float]:
        metrics = {}
        for key, value in self.metrics.items():
            metrics[f"{key}_mean"] = np.mean(value).item()
            metrics[f"{key}_sum"] = np.sum(value).item()

        return metrics
