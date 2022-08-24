import collections
import time
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Tuple
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torchvision.ops import SqueezeExcitation

from common.distributed import print_once
from nvae.distributions import LogisticMixture


def kl_loss(q: Normal, p: Normal) -> torch.Tensor:
    scale_ratio = q.stddev / p.stddev
    scaled_mean_ratio = (q.mean - p.mean) / p.stddev
    return 0.5 * (scale_ratio ** 2 + scaled_mean_ratio ** 2) - 0.5 - torch.log(scale_ratio)


class EncoderBlock(nn.Module):
    """
    An encoder block as depicted in section 3.1.2 of https://arxiv.org/pdf/2007.03898.pdf.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.se = SqueezeExcitation(input_channels=out_channels, squeeze_channels=max(out_channels // 16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.residual(x) + self.block(x))


class EncoderCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.cell = nn.Sequential(
            EncoderBlock(in_channels=in_channels, out_channels=out_channels, stride=stride),
            EncoderBlock(in_channels=out_channels, out_channels=out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell(x)


class DecoderBlock(nn.Module):
    """
    A decoder cell as depicted in section 3.1.1 of https://arxiv.org/pdf/2007.03898.pdf.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        extension_factor: int,
        upsample_fn: nn.Module = nn.Identity(),
    ):
        super().__init__()

        extended_channels = in_channels * extension_factor

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Upsample(scale_factor=2.0),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            )

        self.cell = nn.Sequential(
            upsample_fn,
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=extended_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=extended_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=extended_channels,
                out_channels=extended_channels,
                kernel_size=5,
                groups=extended_channels,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=extended_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=extended_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.se = SqueezeExcitation(input_channels=out_channels, squeeze_channels=max(out_channels // 16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.residual(x) + self.cell(x))


class ConditionalCombinerCell(nn.Module):
    """
    This cell is responsible for creating the input state for Q(z_i | x, z_{<i}) by merging the encoder state
    for x (after going through n-i encoder steps) and the decoder state after i decoder steps. The output of this cell
    is later passed to a "sampler" conv layer which outputs mean and log std for sampling z_i.

    The respective encoder and decoder steps should produce states at the same resolution and number of channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, encoder_state: torch.Tensor, decoder_state: torch.Tensor) -> torch.Tensor:
        return encoder_state + self.conv(decoder_state)


class LatentCombinerCell(nn.Module):
    """
    This cell is responsible for combining a sampled latent variable z_i ~ Q(z_i | x, z_{<i}) with the decoder
    state after i decoder steps.

    The z_i and the decoder state have the same resolution.
    """
    def __init__(self, channels: int, latent_dim: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=channels + latent_dim, out_channels=channels, kernel_size=1)

    def forward(self, decoder_state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([decoder_state, latent], dim=1))


class Preprocessor(nn.Module):
    def __init__(self, in_channels: int, num_cells: int):
        super().__init__()

        self.layers = nn.Sequential(
            # Increase number of channels from 3 (RGB).
            nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, padding=1, bias=True),
        )
        for i in range(num_cells-1):
            self.layers.append(EncoderCell(in_channels=in_channels, out_channels=in_channels, stride=1))
        self.layers.append(EncoderCell(in_channels=in_channels, out_channels=2 * in_channels, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map images to [-1, 1].
        return self.layers(2 * x - 1)


class Postprocessor(nn.Module):
    def __init__(self, in_channels: int, num_cells: int):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(num_cells-1):
            self.layers.append(DecoderBlock(in_channels=in_channels, out_channels=in_channels, extension_factor=3))
        self.layers.append(
            DecoderBlock(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                extension_factor=3,
                upsample_fn=nn.UpsamplingNearest2d(scale_factor=2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ImageConditional(nn.Module):
    def __init__(self, in_channels: int, num_mixtures: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=10 * num_mixtures, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_latent_groups: int,
        num_cells_per_group: int,
        resolution_group_offsets: FrozenSet[int],
    ):
        super().__init__()

        self.groups = nn.ModuleList()
        self.tracked_group_ids = set()
        channels = in_channels
        for i in range(num_latent_groups):
            group = nn.Sequential()
            for _ in range(num_cells_per_group):
                group.append(EncoderCell(in_channels=channels, out_channels=channels, stride=1))

            self.tracked_group_ids.add(len(self.groups))
            self.groups.append(group)

            if i in resolution_group_offsets:
                self.groups.append(EncoderCell(in_channels=channels, out_channels=2*channels, stride=2))
                channels *= 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        encoded_x = []
        for i, m in enumerate(self.groups):
            x = m(x)
            if i in self.tracked_group_ids:
                encoded_x.append(x)

        return x, list(reversed(encoded_x))


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_resolution: int,
        num_latent_groups: int,
        num_cells_per_group: int,
        resolution_group_offsets: FrozenSet[int],
        latent_dim: int,
    ):
        super().__init__()

        self.decoder_cells, self.resolution_cells = self._create_decoder(
            in_channels=in_channels,
            num_latent_groups=num_latent_groups,
            num_cells_per_group=num_cells_per_group,
            resolution_group_offsets=resolution_group_offsets,
        )

        self.decoder_state = nn.Parameter(
            torch.rand(1, in_channels, in_resolution, in_resolution),
            requires_grad=True,
        )

        self.encoder_samplers = self._create_samplers(
            num_latent_groups=num_latent_groups,
            latent_dim=latent_dim,
            in_channels=in_channels,
            resolution_group_offsets=resolution_group_offsets,
            kernel_size=3,
            padding=1,
            activation_fn=nn.Identity,
        )
        self.decoder_samplers = self._create_samplers(
            num_latent_groups=num_latent_groups,
            latent_dim=latent_dim,
            in_channels=in_channels,
            resolution_group_offsets=resolution_group_offsets,
            kernel_size=1,
            padding=0,
            activation_fn=nn.ELU,
        )

        self.conditional_combiners = self._create_conditional_combiners(
            in_channels=in_channels,
            num_latent_groups=num_latent_groups,
            resolution_offsets=resolution_group_offsets,
        )
        self.latent_combiners = self._create_latent_combiners(
            in_channels=in_channels,
            num_latent_groups=num_latent_groups,
            latent_dim=latent_dim,
            resolution_offsets=resolution_group_offsets,
        )

    def _create_decoder(
        self,
        in_channels: int,
        num_latent_groups: int,
        num_cells_per_group: int,
        resolution_group_offsets: FrozenSet[int],
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        decoder_cells = nn.ModuleList()
        resolution_cells = nn.ModuleList()
        channels = in_channels
        for offset in range(num_latent_groups):
            group = nn.Sequential()
            for _ in range(num_cells_per_group):
                group.append(DecoderBlock(in_channels=channels, out_channels=channels, extension_factor=6))
            decoder_cells.append(group)

            if offset in resolution_group_offsets:
                resolution_cells.append(
                    DecoderBlock(
                        in_channels=channels,
                        out_channels=channels // 2,
                        extension_factor=6,
                        upsample_fn=nn.Upsample(scale_factor=2.0),
                    )
                )
                channels //= 2
            else:
                resolution_cells.append(nn.Identity())

        return decoder_cells, resolution_cells

    def _create_samplers(
        self,
        num_latent_groups: int,
        latent_dim: int,
        in_channels: int,
        resolution_group_offsets: FrozenSet[int],
        kernel_size: int,
        padding: int,
        activation_fn: Type[nn.Module] = nn.Identity,
    ) -> nn.ModuleList:
        channels = in_channels
        samplers = nn.ModuleList()
        for group_id in range(num_latent_groups):
            samplers.append(
                nn.Sequential(
                    activation_fn(),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=2 * latent_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                )
            )
            if group_id in resolution_group_offsets:
                channels //= 2

        return samplers

    def _create_conditional_combiners(
        self,
        in_channels: int,
        num_latent_groups: int,
        resolution_offsets: FrozenSet[int],
    ) -> nn.ModuleList:
        ret = nn.ModuleList()
        channels = in_channels
        for i in range(num_latent_groups):
            ret.append(ConditionalCombinerCell(channels=channels))
            if i in resolution_offsets:
                channels //= 2
        return ret

    def _create_latent_combiners(
        self,
        in_channels: int,
        latent_dim: int,
        num_latent_groups: int,
        resolution_offsets: FrozenSet[int],
    ) -> nn.ModuleList:
        ret = nn.ModuleList()
        channels = in_channels
        for i in range(num_latent_groups):
            ret.append(LatentCombinerCell(channels=channels, latent_dim=latent_dim))
            if i in resolution_offsets:
                channels //= 2
        return ret

    def forward(
        self,
        encoder_final_state: torch.Tensor,
        encoder_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        mean, log_std = torch.chunk(self.encoder_samplers[0](encoder_final_state), dim=1, chunks=2)
        # TODO(pauldb): Revisit clamping etc.
        q_dist = Normal(loc=mean, scale=torch.exp(log_std))
        latent = q_dist.rsample()
        log_qs = [q_dist.log_prob(latent)]

        p_dist = Normal(loc=torch.zeros_like(mean), scale=torch.ones_like(log_std))
        log_ps = [p_dist.log_prob(latent)]
        kl_divs = [kl_loss(q_dist, p_dist)]

        batch_size = encoder_final_state.size()[0]
        x = self.decoder_cells[0](self.decoder_state.expand(batch_size, -1, -1, -1))
        x = self.latent_combiners[0](decoder_state=x, latent=latent)

        for idx in range(1, len(self.decoder_cells)):
            x = self.decoder_cells[idx](x)

            p_mean, p_log_std = torch.chunk(self.decoder_samplers[idx](x), dim=1, chunks=2)
            # TODO(pauldb): More clamping.
            std = torch.exp(p_log_std)
            mask = std <= 0
            assert torch.min(std).item() > 0 # , std[mask]
            p_dist = Normal(loc=p_mean, scale=torch.exp(p_log_std))

            q_state = self.conditional_combiners[idx](encoder_state=encoder_states[idx-1], decoder_state=x)
            q_mean, q_log_std = torch.chunk(self.encoder_samplers[idx](q_state), dim=1, chunks=2)
            q_dist = Normal(loc=q_mean, scale=torch.exp(q_log_std))
            latent = q_dist.rsample()

            kl_divs.append(kl_loss(q_dist, p_dist))
            log_qs.append(q_dist.log_prob(latent))
            log_qs.append(p_dist.log_prob(latent))

            x = self.latent_combiners[idx](decoder_state=x, latent=latent)

            x = self.resolution_cells[idx](x)

        return x, kl_divs, log_qs, log_ps


class NVAE(nn.Module):
    def __init__(
        self,
        num_latent_groups: int = 36,
        latent_dim: int = 20,
        num_encoder_channels: int = 30,
        num_preprocess_cells: int = 2,
        num_postprocess_cells: int = 2,
        num_encoder_cells: int = 2,
        num_decoder_cells: int = 2,
        num_output_mixtures: int = 10,
        # TODO(pauldb): Should I include 36 too?
        encoder_resolution_group_offsets: FrozenSet[int] = frozenset([15, 23, 27, 31]),
        power_method_steps: int = 4,
    ):
        super().__init__()

        self.preprocessor = Preprocessor(
            in_channels=num_encoder_channels,
            num_cells=num_preprocess_cells,
        )

        self.encoder = Encoder(
            in_channels=2 * num_encoder_channels,
            num_latent_groups=num_latent_groups,
            num_cells_per_group=num_encoder_cells,
            resolution_group_offsets=encoder_resolution_group_offsets,
        )

        bottleneck_scale_factor = 2 ** (len(encoder_resolution_group_offsets) + 1)
        bottleneck_channels = bottleneck_scale_factor * num_encoder_channels
        decoder_resolution_group_offsets = frozenset(
            [num_latent_groups - 1 - x for x in encoder_resolution_group_offsets]
        )

        self.mid_processor = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=1),
            nn.ELU(),
        )

        self.decoder = Decoder(
            in_channels=bottleneck_channels,
            in_resolution=256 // bottleneck_scale_factor,
            num_latent_groups=num_latent_groups,
            num_cells_per_group=num_decoder_cells,
            resolution_group_offsets=decoder_resolution_group_offsets,
            latent_dim=latent_dim,
        )

        self.postprocessor = Postprocessor(
            in_channels=2 * num_encoder_channels,
            num_cells=num_postprocess_cells,
        )

        self.num_output_mixtures = num_output_mixtures
        self.image_conditional = ImageConditional(in_channels=num_encoder_channels, num_mixtures=num_output_mixtures)

        self.cuda()

        # Parameters needed for the power iteration method to compute the spectral regularization loss.
        self.power_method_steps = power_method_steps
        self.u = nn.ParameterDict()
        self.v = nn.ParameterDict()
        grouped_weights = self._group_weights()
        for key, weight in grouped_weights.items():
            batch, rows, cols = key.split("_")
            self.u[key] = nn.Parameter(torch.randn(int(batch), int(rows), device="cuda"), requires_grad=False)
            self.v[key] = nn.Parameter(torch.randn(int(batch), int(cols), device="cuda"), requires_grad=False)

        print_once(f"Executing {self.power_method_steps * 10} power method steps to warm up spectral normalization...")
        start_time = time.time()
        self._power_method(grouped_weights, steps=self.power_method_steps * 10)
        print_once(f"Spectral normalization warm up took {time.time() - start_time} seconds")

    def batch_norm_loss(self) -> torch.Tensor:
        losses = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) and module.affine:
                losses.append(torch.max(torch.abs(module.weight)))

        return sum(losses)

    def _group_weights(self) -> Dict[str, torch.Tensor]:
        grouped_weights = collections.defaultdict(list)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                weight_mat = m.weight.view(m.weight.size(0), -1)
                grouped_weights[weight_mat.size()].append(weight_mat)

        ret = {}
        for weights in grouped_weights.values():
            weights = torch.stack(weights, dim=0)
            key = "_".join(str(d) for d in weights.size())
            ret[key] = weights

        return ret

    def _power_method(self, grouped_weights: Dict[str, torch.Tensor], steps: int):
        with torch.no_grad():
            for key, weights in grouped_weights.items():
                for _ in range(steps):
                    self.v[key] = F.normalize(
                        # (B x R) * (B x R x C) => (B x 1 x R) * (B x R x C) => (B x 1 x C) => B x C
                        torch.bmm(self.u[key].unsqueeze(dim=1), weights).squeeze(dim=1),
                        dim=1,
                        eps=1e-3
                    )
                    self.u[key] = F.normalize(
                        # (B x R x C) * (B x C) => (B x R x C) * (B x C x 1) => (B x R x 1) => B x R
                        torch.bmm(weights, self.v[key].unsqueeze(dim=2)).squeeze(dim=2),
                        dim=1,
                        eps=1e-3,
                    )

    def spectral_loss(self) -> torch.Tensor:
        grouped_weights = self._group_weights()
        self._power_method(grouped_weights, steps=self.power_method_steps)

        losses = []
        for key, weights in grouped_weights.items():
            singular_value = torch.bmm(self.u[key].unsqueeze(dim=1), torch.bmm(weights, self.v[key].unsqueeze(dim=2)))
            losses.append(torch.mean(singular_value))

        return sum(losses)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        images = images.to(device="cuda")

        x = self.preprocessor(images)
        x, encoder_states = self.encoder(x)
        x = self.mid_processor(x)

        x, kl_divs, log_qs, log_ps = self.decoder(encoder_final_state=x, encoder_states=encoder_states)

        x = self.postprocessor(x)
        x = self.image_conditional(x)

        dist = LogisticMixture(num_mixtures=self.num_output_mixtures, params=x)

        reconstruction_loss = torch.mean(dist.log_p(images))
        total_kl_loss = torch.mean(torch.stack([torch.mean(kl_div) for kl_div in kl_divs]))
        bn_loss = self.batch_norm_loss()
        spectral_loss = self.spectral_loss()

        losses = {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": total_kl_loss,
            "bn_loss": bn_loss,
            "spectral_loss": spectral_loss,
        }

        return {
            **losses,
            "loss": sum(losses.values()),
        }
