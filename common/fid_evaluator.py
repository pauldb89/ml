import time
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.distributed import all_gather

from common.distributed import print_once
from common.distributed import world_size


class FIDEvaluator:
	"""
	Compute Fréchet inception distance (FID) as described in https://arxiv.org/pdf/1706.08500.pdf.
	"""
	def __init__(self, data_loader: Iterable[torch.Tensor]):
		self.inception_v3 = InceptionV3().cuda()
		self.inception_v3.eval()

		self.num_ref_samples = len(data_loader.dataset)
		self.ref_mean, self.ref_covariance = self._compute_statistics(data_loader, num_samples=self.num_ref_samples)

	def _compute_statistics(
		self,
		data_loader: Iterable[torch.Tensor],
		num_samples: int,
		log_progress_every_n_steps: int = 10,
	) -> Tuple[np.ndarray, np.ndarray]:
		num_samples = num_samples // world_size()

		start_time = time.time()
		with torch.inference_mode():
			per_rank_activations = []
			current_samples = 0
			num_steps = 0
			generator = iter(data_loader)
			while current_samples < num_samples:
				try:
					images = next(generator)
				except StopIteration:
					if current_samples <= 0.9 * num_samples:
						print_once(
							f"Error: FID evaluator expected {num_samples}, "
							f"but only {current_samples} were provided!"
						)
						raise
					else:
						print_once(
							f"Warning: FID evaluator expected {num_samples}, "
							f"but only {current_samples} were provided!"
						)
						break

				per_rank_activations.extend(self.inception_v3(images.cuda()))
				current_samples += images.size(0)
				num_steps += 1
				if num_steps % log_progress_every_n_steps == 0:
					print_once(f"{current_samples}/{num_samples} samples evaluated")

			per_rank_activations = torch.flatten(torch.cat(per_rank_activations, dim=0), start_dim=1)
			all_activations = [torch.zeros_like(per_rank_activations) for _ in range(world_size())]
			all_gather(all_activations, per_rank_activations)

			all_activations = torch.cat(all_activations, dim=0)

		mean = torch.mean(all_activations, dim=0).cpu().numpy()
		covariance = torch.cov(all_activations.t()).cpu().numpy()

		print_once(f"Computing mean and covariance for FID score took {time.time() - start_time} seconds...")
		return mean, covariance

	def fid_score(
		self,
		sampling_iterator: Iterable[torch.Tensor],
		num_samples: Optional[int] = None,
	) -> Optional[float]:
		print_once("Computing FID score...")
		start_time = time.time()
		mean, covariance = self._compute_statistics(sampling_iterator, num_samples=num_samples or self.num_ref_samples)

		assert mean.shape == self.ref_mean.shape, (mean.shape, self.ref_mean.shape)
		assert covariance.shape == self.ref_covariance.shape, (covariance.shape, self.ref_covariance.shape)

		try:
			fid_score = calculate_frechet_distance(
				mu1=self.ref_mean,
				sigma1=self.ref_covariance,
				mu2=mean,
				sigma2=covariance,
			)
		except ValueError as e:
			print_once(f"Exception while computing FID score: {e}")
			fid_score = None

		print_once(f"Computing FID score took {time.time() - start_time} seconds...")
		return fid_score
