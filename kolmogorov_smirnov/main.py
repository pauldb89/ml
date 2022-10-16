import math
import os
from argparse import ArgumentParser
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from common.wandb import wandb_log


def compute_poly_pdf(x: np.ndarray, num_draws: int) -> np.ndarray:
    return num_draws * (1 - x) ** (num_draws - 1)


def compute_exp_pdf(x: np.ndarray, gamma: float) -> np.ndarray:
    return gamma * np.exp(-gamma * x)


def compute_empirical_pdf(
    num_samples: int,
    num_draws: int,
    column_index: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert 0 <= column_index < num_draws - 1

    data = np.random.uniform(low=0, high=1, size=(num_samples, num_draws))
    data = np.sort(data, axis=1)

    delta = data[:, column_index + 1] - data[:, column_index]
    assert 0 <= np.min(delta) and np.max(delta) <= 1

    return np.histogram(delta, bins=num_bins, density=True)


def plot_pdfs(
    x: np.ndarray,
    empirical_pdf: np.ndarray,
    pdfs: Dict[str, np.ndarray],
    output_dir: str,
) -> None:
    plt.figure(figsize=(10, 10))
    plt.plot(x, empirical_pdf, label="empirical")
    for key, pdf in pdfs.items():
        plt.plot(x, pdf, label=key)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_dir, "viz.png"))

    for i in range(len(x)):
        values = {"x": x[i], "empirical": empirical_pdf[i]}
        for key, pdf in pdfs.items():
            values[key] = pdf[i]
        wandb_log(values)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute KL divergence KL(p || q).

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.
    """
    return np.sum(p * np.log(p / q + eps)).item()


def cdf(pdf: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    return np.cumsum(pdf * (intervals[1:] - intervals[:-1]))


def ks_test(
    p: np.ndarray,
    q: np.ndarray,
    intervals: np.ndarray,
    p_value: float = 0.05,
):
    """
    Kolmogorov-Smirnov Test:
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.

    The hypothesis is that the empirical distribution p is the same as the
    distribution q. This hypothesis can be rejected with a certain p-value
    if the maximum absolute distance between the CDFs is above a threshold
    that depends on sample size and p-value.

    This method returns False if the test indicates that the distributions
    are likely not the same.
    """
    thresholds = {
        0.2: 1.073,
        0.15: 1.138,
        0.10: 1.224,
        0.05: 1.358,
        0.025: 1.48,
        0.01: 1.628,
        0.005: 1.731,
        0.001: 1.949
    }
    cp = cdf(p, intervals)
    cq = cdf(q, intervals)
    num_bins = len(intervals) - 1
    max_diff = np.max(np.abs(cp - cq))
    threshold = thresholds[p_value]
    return max_diff <= threshold * math.sqrt(2 / num_bins)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--num_bins", type=int, default=250, help="Number of histogram bins for empirical distribution"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1_000_000, help="Number of samples to estimate empirical distribution"
    )
    parser.add_argument("--num_draws", type=int, default=10, help="How many draws to take from U[0, 1]")
    parser.add_argument(
        "--column_index", type=int, default=5, help="Compute the difference between which consecutive draws"
    )
    parser.add_argument(
        "--exp_params",
        type=str,
        default="1,5,10,50",
        help="Lambda parameters to try for fitting exponential distributions",
    )
    parser.add_argument("--seed", type=int, default=5, help="Random seed for deterministic results")
    parser.add_argument(
        "--output_dir", type=str, default="/models/kolmogorov_smirnov", help="Directory for visualizations"
    )
    args = parser.parse_args()

    wandb_init("kolmogorov-smirnov", dir=WANDB_DIR)
    wandb_config_update(args)

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    empirical_pdf, intervals = compute_empirical_pdf(
        num_samples=args.num_samples,
        num_draws=args.num_draws,
        column_index=args.column_index,
        num_bins=args.num_bins,
    )
    x = (intervals[1:] + intervals[:-1]) / 2

    pdfs = {"poly": compute_poly_pdf(x, num_draws=args.num_draws)}
    for gamma in map(int, args.exp_params.split(",")):
        pdfs[f"exp_{gamma}"] = compute_exp_pdf(x, gamma)
    plot_pdfs(x, empirical_pdf, pdfs, output_dir=args.output_dir)

    for key, pdf in pdfs.items():
        print(f"KL divergence between empirical and "
              f"{key} distribution is {kl_div(empirical_pdf, pdf)}")

    for key, pdf in pdfs.items():
        ret = ks_test(p=empirical_pdf, q=pdf, intervals=intervals)
        print(f"Empirical distribution is likely the same as {key} distribution: {ret}")


if __name__ == "__main__":
    main()
