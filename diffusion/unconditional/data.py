from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
from torchvision import transforms

from common.data import ImageDataset


def create_train_data_loader(
	data_dir: str,
	batch_size: int,
	max_steps: int,
	resolution: int,
	num_workers: int = 5,
) -> DataLoader:
	dataset = ImageDataset(
		root=data_dir,
		transform=transforms.Compose([
			transforms.Resize(resolution),
			transforms.RandomHorizontalFlip(),
			transforms.CenterCrop(resolution),
			transforms.ToTensor(),
		]),
	)
	return DataLoader(
		dataset=dataset,
		sampler=RandomSampler(data_source=dataset, replacement=False, num_samples=max_steps * batch_size),
		batch_size=batch_size,
		drop_last=True,
		num_workers=num_workers,
	)


def create_eval_data_loader(
	data_dir: str,
	batch_size: int,
	resolution: int,
	drop_last: bool = False,
	num_workers: int = 5,
) -> DataLoader:
	dataset = ImageDataset(
		root=data_dir,
		transform=transforms.Compose([
			transforms.Resize(resolution),
			transforms.CenterCrop(resolution),
			transforms.ToTensor(),
		]),
	)
	return DataLoader(
		dataset=dataset,
		sampler=DistributedSampler(dataset=dataset, shuffle=False),
		batch_size=batch_size,
		drop_last=drop_last,
		num_workers=num_workers,
	)
