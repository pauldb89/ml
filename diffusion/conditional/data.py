from __future__ import annotations

import random
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
from torchvision.datasets import CocoCaptions
from torchvision.transforms import transforms
from transformers import AutoTokenizer

from common.consts.coco_consts import EVAL_CAPTIONS_FILE
from common.consts.coco_consts import EVAL_ROOT_DIR
from common.consts.coco_consts import TRAIN_CAPTIONS_FILE
from common.consts.coco_consts import TRAIN_ROOT_DIR


class TokenizeCaption:
	def __init__(self, text_encoder: str):
		self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

	def __call__(self, captions: List[str]) -> Dict[str, Any]:
		raw_caption = captions[random.randrange(0, len(captions))]
		return {"raw_caption": raw_caption, **self.tokenizer(raw_caption)}


class Batch(NamedTuple):
	images: Optional[torch.Tensor]
	token_ids: torch.Tensor
	token_masks: torch.Tensor
	raw_captions: List[str]

	def to(self, device: torch.device) -> Batch:
		return Batch(
			images=self.images.to(device) if self.images is not None else None,
			token_ids=self.token_ids.to(device),
			token_masks=self.token_masks.to(device),
			raw_captions=self.raw_captions,
		)

	def __len__(self) -> int:
		return len(self.raw_captions)

	def __getitem__(self, key: Union[int, slice]) -> Batch:
		return Batch(
			images=self.images[key] if self.images is not None else None,
			token_ids=self.token_ids[key],
			token_masks=self.token_masks[key],
			raw_captions=self.raw_captions[key],
		)


def collate(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Batch:
	images, captions = zip(*batch)

	token_ids = []
	attention_masks = []
	raw_captions = []
	for caption in captions:
		token_ids.append(torch.tensor(caption["input_ids"]))
		attention_masks.append(torch.tensor(caption["attention_mask"]))
		raw_captions.append(caption["raw_caption"])

	return Batch(
		images=torch.stack(images, dim=0),
		token_ids=pad_sequence(token_ids, batch_first=True),
		token_masks=pad_sequence(attention_masks, batch_first=True),
		raw_captions=raw_captions,
	)


def create_train_data_loader(
	batch_size: int,
	max_steps: int,
	resolution: int,
	text_encoder: str,
	num_workers: int = 5,
) -> DataLoader:
	dataset = CocoCaptions(
		root=TRAIN_ROOT_DIR,
		annFile=TRAIN_CAPTIONS_FILE,
		transform=transforms.Compose([
			transforms.Resize(resolution),
			transforms.CenterCrop(resolution),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		]),
		target_transform=TokenizeCaption(text_encoder),
	)

	return DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		sampler=RandomSampler(data_source=dataset, replacement=False, num_samples=max_steps * batch_size),
		drop_last=True,
		num_workers=num_workers,
		collate_fn=collate,
	)


def create_eval_data_loader(
	batch_size: int,
	resolution: int,
	text_encoder: str,
	num_workers: int = 5,
) -> DataLoader:
	dataset = CocoCaptions(
		root=EVAL_ROOT_DIR,
		annFile=EVAL_CAPTIONS_FILE,
		transform=transforms.Compose([
			transforms.Resize(resolution),
			transforms.CenterCrop(resolution),
			transforms.ToTensor(),
		]),
		target_transform=TokenizeCaption(text_encoder),
	)

	return DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		sampler=DistributedSampler(dataset=dataset, shuffle=False),
		drop_last=False,
		num_workers=num_workers,
		collate_fn=collate,
	)
