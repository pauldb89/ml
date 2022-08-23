from torch import nn
from torch.utils.data import DataLoader


def evaluate(step: str, model: nn.Module, data_loader: DataLoader) -> None:
    print(f"Evaluation for {step}")
