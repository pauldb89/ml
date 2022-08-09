import random
import time
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Set

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import VGG19_Weights
from torchvision.models import vgg19

from common.consts import IMAGENET_MEAN
from common.consts import IMAGENET_STD
from common.distributed import is_root_process
from common.distributed import print_once
from neural_style_transfer.data import Batch


class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, content_map: torch.Tensor, style_map: torch.Tensor) -> torch.Tensor:
        assert content_map.size() == style_map.size()

        batch_size, channels, height, width = content_map.size()

        content_map = content_map.view(batch_size, channels, -1)
        style_map = style_map.view(batch_size, channels, -1)

        content_means = torch.mean(content_map, dim=-1, keepdim=True)
        content_stds = torch.std(content_map, dim=-1, keepdim=True)
        style_means = torch.mean(style_map, dim=-1, keepdim=True)
        style_stds = torch.std(style_map, dim=-1, keepdim=True)

        adaptive_map = ((content_map - content_means) / (content_stds + self.eps)) * style_stds + style_means

        return adaptive_map.view(batch_size, channels, height, width)


class VGGImageEncoder(nn.Module):
    def __init__(self, feature_names: Set[str]):
        super().__init__()
        self.feature_names = feature_names
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ret = {}
        block_id = 1
        layer_id = 1
        for m in self.model.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                block_id += 1
                layer_id = 1
            elif isinstance(m, nn.Conv2d) and layer_id == 1:
                key = f"layer{block_id}"
                if key in self.feature_names:
                    ret[key] = x
                layer_id += 1

        assert len(ret) == len(self.feature_names), (ret.keys(), self.feature_names)

        return ret


class VGGImageDecoder(nn.Module):
    def __init__(self, interpolation: str = "nearest"):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode=interpolation),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode=interpolation),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode=interpolation),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.features(feature_map)


class StyleLoss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.size() == y.size()
        batch_size, channels, _, _ = x.size()

        x = x.view(batch_size, channels, -1)
        y = y.view(batch_size, channels, -1)

        mean_loss = F.mse_loss(torch.mean(x, dim=-1), torch.mean(y, dim=-1))
        std_loss = F.mse_loss(torch.std(x, dim=-1), torch.std(y, dim=-1))

        return mean_loss + std_loss


class StyleTransfer(nn.Module):
    CONTENT_LOSS_KEY: str = "content_loss"
    STYLE_LOSS_KEY: str = "style_loss"
    LOSS_KEYS: List[str] = [CONTENT_LOSS_KEY, STYLE_LOSS_KEY]

    def __init__(
        self,
        content_feature_name: str,
        style_feature_names: List[str],
        style_weight: float = 1.0,
        interpolation: str = "nearest",
    ):
        super().__init__()

        self.content_feature_name = content_feature_name
        self.style_feature_names = style_feature_names
        self.style_weight = style_weight

        self.encoder = VGGImageEncoder(feature_names={content_feature_name, *style_feature_names})
        self.ada_in = AdaptiveInstanceNormalization()
        self.decoder = VGGImageDecoder(interpolation=interpolation)

        self.transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.style_loss = StyleLoss()

        self.original_encoder_time = 0
        self.ada_in_time = 0
        self.decoder_time = 0
        self.post_encoder_time = 0

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        batch = batch.to(torch.device("cuda"))

        start_time = time.time()
        content_feature_maps = self.encoder(batch.content_images)
        style_feature_maps = self.encoder(batch.style_images)
        end_time = time.time()
        self.original_encoder_time += end_time - start_time

        start_time = end_time
        adaptive_map = self.ada_in(
            content_feature_maps[self.content_feature_name],
            style_feature_maps[self.content_feature_name],
        )
        end_time = time.time()
        self.ada_in_time += end_time - start_time

        start_time = end_time
        decoded_images = self.decoder(adaptive_map)
        end_time = time.time()
        self.decoder_time += end_time - start_time

        start_time = end_time
        transformed_images = self.transform(decoded_images)

        decoded_feature_maps = self.encoder(transformed_images)
        end_time = time.time()
        self.post_encoder_time += end_time - start_time

        content_loss = F.mse_loss(adaptive_map, decoded_feature_maps[self.content_feature_name])
        style_loss = sum([
            self.style_loss(style_feature_maps[feature_name], decoded_feature_maps[feature_name])
            for feature_name in self.style_feature_names
        ])

        return {
            "content_loss": content_loss,
            "style_loss": style_loss,
            "loss": content_loss + self.style_weight * style_loss,
        }

    def eval_forward(self, content_image: torch.Tensor, style_image: torch.Tensor) -> torch.Tensor:
        content_feature_maps = self.encoder(content_image.cuda())
        style_feature_maps = self.encoder(style_image.cuda())
        adaptive_map = self.ada_in(
            content_feature_maps[self.content_feature_name],
            style_feature_maps[self.content_feature_name],
        )
        return self.decoder(adaptive_map)

    def summarize(self, step: int, epoch: int, metrics: Dict[str, torch.Tensor]) -> None:
        formatted_metrics = "\t".join([f"{k}: {metrics[k].item()}" for k in self.LOSS_KEYS])
        print(f"Step {step}:\t{formatted_metrics}")
        # print(f"Original encoder time {self.original_encoder_time}")
        # print(f"Ada in time {self.ada_in_time}")
        # print(f"Decoder time {self.decoder_time}")
        # print(f"Post encoder time {self.post_encoder_time}")
