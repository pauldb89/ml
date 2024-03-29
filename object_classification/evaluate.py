from typing import Optional

import torch
from torch import nn
from torch.distributed import get_rank
from torch.utils.data import DataLoader
from torch.nn import functional as F

from common.distributed import print_once


def evaluate(
    step: int,
    eval_swag: bool,
    model: nn.Module,
    data_loader: DataLoader,
    log_progress_every_n_steps: Optional[int] = 30,
) -> None:
    total = torch.tensor(0.0, device=torch.device("cuda"))
    top_1_matches = torch.tensor(0.0, device=torch.device("cuda"))
    top_5_matches = torch.tensor(0.0, device=torch.device("cuda"))
    loss = torch.tensor(0.0, device=torch.device("cuda"))

    print_once("Evaluating model....")
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(data_loader, start=1):
            batch = batch.to(torch.device("cuda"))

            predictions = model.eval_forward(batch.images)
            _, indices = torch.topk(predictions, k=5, dim=1)

            total += torch.numel(batch.classes)
            top_1_matches += (batch.classes == indices[:, 0]).sum()
            top_5_matches += (torch.unsqueeze(batch.classes, dim=1) == indices).sum()
            loss += F.nll_loss(torch.log(predictions), batch.classes, reduction="sum")

            if log_progress_every_n_steps is not None and batch_idx % log_progress_every_n_steps == 0:
                print(f"{total} examples processed on rank {get_rank()}...")

        model.train()

    torch.distributed.all_reduce(top_1_matches)
    torch.distributed.all_reduce(top_5_matches)
    torch.distributed.all_reduce(total)
    torch.distributed.all_reduce(loss)

    formatted_step = step if not eval_swag else f"{step} SWAG"
    print_once(f"Step {formatted_step}: Top-1 accuracy {100 * (top_1_matches / total).item():.1f}%")
    print_once(f"Step {formatted_step}: Top-5 accuracy {100 * (top_5_matches / total).item():.1f}%")
    print_once(f"Step {formatted_step}: Eval Loss {(loss / total).item():.5f}")
