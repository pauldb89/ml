from typing import Optional

import torch
from torch.distributed import get_rank
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.nn import functional as F

from object_classification.distributed_utils import print_once


def evaluate(
    step: int,
    model: DistributedDataParallel,
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

            predictions = model.module.eval_forward(batch.images)
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

    print_once(f"Step {step}: Top-1 accuracy {100 * (top_1_matches / total).item():.1f}%")
    print_once(f"Step {step}: Top-5 accuracy {100 * (top_5_matches / total).item():.1f}%")
    print_once(f"Step {step}: Eval Loss {(loss / total).item():.5f}")
