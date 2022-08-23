from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.transforms import transforms

from common.data import ImageDataset


def create_data_loader(root: str, batch_size: int, is_train: bool, num_workers: int = 5) -> DataLoader:
    dataset = ImageDataset(root=root, transform=transforms.ToTensor())
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset=dataset, shuffle=is_train),
        num_workers=num_workers,
        drop_last=is_train,
    )
