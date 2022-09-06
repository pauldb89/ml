import collections
import time
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torchvision.ops import SqueezeExcitation

from common.distributed import print_once
from nvae.distributions import LogisticMixture
from nvae.distributions import create_normal


def kl_loss(q: Normal, p: Normal) -> torch.Tensor:
    scale_ratio = q.stddev / p.stddev
    scaled_mean_ratio = (q.mean - p.mean) / p.stddev
    return 0.5 * (scale_ratio ** 2 + scaled_mean_ratio ** 2) - 0.5 - torch.log(scale_ratio)


def conv_l2_norm(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norms for each output channel for a 2D convolution weight matrix. The output matrix is reshaped to
    a 4D tensor to facilitate division operations with the original weight matrix to normalize it.
    """
    out_channels, _, _, _ = weight.size()
    return torch.norm(weight.view(out_channels, -1), dim=1, p=2).view(out_channels, 1, 1, 1)


class ConvBlock(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        weight_norm: bool = True,
        eps: float = 1e-5,
    ):
        """
        2D convolution block with optional weight normalization.

        If weight normalization is enabled, the weights for each output channel are L2 normalized and the weights
        are additionally scaled by a learned normalization parameter (in log scale).
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        self.eps = eps

        self.log_weight_norm = None
        if weight_norm:
            self.log_weight_norm = nn.Parameter(torch.log(conv_l2_norm(self.weight) + 1e-2), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_weight_norm is not None:
            weight = torch.exp(self.log_weight_norm) * self.weight / (conv_l2_norm(self.weight) + self.eps)
        else:
            weight = self.weight

        return F.conv2d(
            input=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.act = nn.SiLU()
        self.conv_1 = ConvBlock(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = ConvBlock(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = ConvBlock(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = ConvBlock(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = self.act(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class EncoderBlock(nn.Module):
    """
    An encoder block as depicted in section 3.1.2 of https://arxiv.org/pdf/2007.03898.pdf.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            # TODO(pauldb): See if reverting this makes any difference.
            # self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            self.residual = FactorizedReduce(C_in=in_channels, C_out=out_channels)

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=0.05),
            nn.SiLU(),
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels, momentum=0.05),
            nn.SiLU(),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        self.se = SqueezeExcitation(input_channels=out_channels, squeeze_channels=max(out_channels // 16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + 0.1 * self.se(self.block(x))


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
                nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True),
                ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            )

        self.cell = nn.Sequential(
            upsample_fn,
            nn.BatchNorm2d(num_features=in_channels, momentum=0.05),
            ConvBlock(
                in_channels=in_channels,
                out_channels=extended_channels,
                kernel_size=1,
                bias=False,
                weight_norm=False,
            ),
            nn.BatchNorm2d(num_features=extended_channels, momentum=0.05),
            nn.SiLU(),
            ConvBlock(
                in_channels=extended_channels,
                out_channels=extended_channels,
                kernel_size=5,
                groups=extended_channels,
                padding=2,
                bias=False,
                weight_norm=False,
            ),
            nn.BatchNorm2d(num_features=extended_channels, momentum=0.05),
            nn.SiLU(),
            ConvBlock(
                in_channels=extended_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                weight_norm=False,
            ),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.05),
        )

        self.se = SqueezeExcitation(input_channels=out_channels, squeeze_channels=max(out_channels // 16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + 0.1 * self.se(self.cell(x))


class ConditionalCombinerCell(nn.Module):
    """
    This cell is responsible for creating the input state for Q(z_i | x, z_{<i}) by merging the encoder state
    for x (after going through n-i encoder steps) and the decoder state after i decoder steps. The output of this cell
    is later passed to a "sampler" conv layer which outputs mean and log std for sampling z_i.

    The respective encoder and decoder steps should produce states at the same resolution and number of channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=1, bias=True, weight_norm=True)

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

        self.conv = ConvBlock(
            in_channels=channels + latent_dim,
            out_channels=channels,
            kernel_size=1,
            bias=True,
            weight_norm=True,
        )

    def forward(self, decoder_state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([decoder_state, latent], dim=1))


class Preprocessor(nn.Module):
    def __init__(self, in_channels: int, num_cells: int):
        super().__init__()

        self.layers = nn.Sequential(
            # Increase number of channels from 3 (RGB).
            ConvBlock(in_channels=3, out_channels=in_channels, kernel_size=3, padding=1, bias=True, weight_norm=True),
        )
        for i in range(num_cells-1):
            self.layers.append(EncoderBlock(in_channels=in_channels, out_channels=in_channels, stride=1))
        self.layers.append(EncoderBlock(in_channels=in_channels, out_channels=2 * in_channels, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map images to [-1, 1].
        return self.layers(2 * x - 1)


class Postprocessor(nn.Module):
    def __init__(self, in_channels: int, num_cells: int):
        super().__init__()

        self.layers = nn.Sequential(
            DecoderBlock(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                extension_factor=3,
                upsample_fn=nn.UpsamplingNearest2d(scale_factor=2),
            ),
        )
        for i in range(num_cells-1):
            self.layers.append(
                DecoderBlock(
                    in_channels=in_channels // 2,
                    out_channels=in_channels // 2,
                    extension_factor=3,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ImageConditional(nn.Module):
    def __init__(self, in_channels: int, num_mixtures: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ELU(),
            ConvBlock(
                in_channels=in_channels,
                out_channels=10 * num_mixtures,
                kernel_size=3,
                padding=1,
                bias=True,
                weight_norm=True,
            ),
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
                group.append(EncoderBlock(in_channels=channels, out_channels=channels, stride=1))

            self.tracked_group_ids.add(len(self.groups))
            self.groups.append(group)

            if i in resolution_group_offsets:
                self.groups.append(EncoderBlock(in_channels=channels, out_channels=2*channels, stride=2))
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

        self.latent_dim = latent_dim
        self.in_resolution = in_resolution

        self.decoder_cells, self.resolution_cells = self._create_decoder(
            in_channels=in_channels,
            num_latent_groups=num_latent_groups,
            num_cells_per_group=num_cells_per_group,
            resolution_group_offsets=resolution_group_offsets,
        )

        self.conditional_combiners = self._create_conditional_combiners(
            in_channels=in_channels,
            num_latent_groups=num_latent_groups,
            resolution_offsets=resolution_group_offsets,
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
        # First sample on the decoder path is drawn from N(0, 1).
        self.decoder_samplers = self._create_samplers(
            num_latent_groups=num_latent_groups,
            latent_dim=latent_dim,
            in_channels=in_channels,
            resolution_group_offsets=resolution_group_offsets,
            kernel_size=1,
            padding=0,
            activation_fn=nn.ELU,
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
                        upsample_fn=nn.UpsamplingNearest2d(scale_factor=2.0),
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
                    ConvBlock(
                        in_channels=channels,
                        out_channels=2 * latent_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=True,
                        weight_norm=True,
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
        q_dist = create_normal(mean=mean, log_std=log_std)
        latent = q_dist.rsample()
        log_qs = [q_dist.log_prob(latent)]

        p_dist = create_normal(mean=torch.zeros_like(mean), log_std=torch.zeros_like(log_std))
        log_ps = [p_dist.log_prob(latent)]
        kl_divs = [kl_loss(q_dist, p_dist)]

        batch_size = encoder_final_state.size()[0]
        x = self.decoder_cells[0](self.decoder_state.expand(batch_size, -1, -1, -1))
        x = self.latent_combiners[0](decoder_state=x, latent=latent)

        for idx in range(1, len(self.decoder_cells)):
            x = self.decoder_cells[idx](x)

            p_mean, p_log_std = torch.chunk(self.decoder_samplers[idx](x), dim=1, chunks=2)
            p_dist = create_normal(mean=p_mean, log_std=p_log_std)

            q_state = self.conditional_combiners[idx](encoder_state=encoder_states[idx], decoder_state=x)
            q_delta_mean, q_delta_log_std = torch.chunk(self.encoder_samplers[idx](q_state), dim=1, chunks=2)
            q_dist = create_normal(mean=p_mean + q_delta_mean, log_std=p_log_std + q_delta_log_std)
            latent = q_dist.rsample()

            kl_divs.append(kl_loss(q_dist, p_dist))
            log_qs.append(q_dist.log_prob(latent))
            log_qs.append(p_dist.log_prob(latent))

            x = self.latent_combiners[idx](decoder_state=x, latent=latent)

            x = self.resolution_cells[idx](x)

        return x, kl_divs, log_qs, log_ps

    def sample(self, batch_size: int) -> torch.Tensor:
        dist = create_normal(
            mean=torch.zeros(batch_size, self.latent_dim, self.in_resolution, self.in_resolution, device="cuda"),
            log_std=torch.zeros(batch_size, self.latent_dim, self.in_resolution, self.in_resolution, device="cuda"),
        )
        latent = dist.sample()
        x = self.decoder_cells[0](self.decoder_state.expand(batch_size, -1, -1, -1))
        x = self.latent_combiners[0](decoder_state=x, latent=latent)

        for idx in range(1, len(self.decoder_cells)):
            x = self.decoder_cells[idx](x)

            mean, log_std = torch.chunk(self.decoder_samplers[idx](x), dim=1, chunks=2)
            dist = create_normal(mean=mean, log_std=log_std)
            latent = dist.sample()

            x = self.latent_combiners[idx](decoder_state=x, latent=latent)
            x = self.resolution_cells[idx](x)

        return x


class LossConfig(NamedTuple):
    kl_const_steps: int
    kl_anneal_steps: int
    kl_min_coeff: float
    reg_weight: float

    def kl_coeff(self, steps: torch.Tensor) -> float:
        return max(min((steps.item() - self.kl_const_steps) / self.kl_anneal_steps, 1.0), self.kl_min_coeff)


class NVAE(nn.Module):
    RECON_LOSS: str = "reconstruction_loss"
    KL_LOSS: str = "kl_loss"
    BN_LOSS: str = "batchnorm_loss"
    SPECTRAL_LOSS: str = "spectral_loss"
    KL_COEFF: str = "kl_coeff"
    LOSS_KEYS: Tuple[str, ...] = (KL_LOSS, BN_LOSS, SPECTRAL_LOSS, RECON_LOSS)

    def __init__(
        self,
        loss_config: LossConfig,
        image_resolution: int = 64,
        num_latent_groups: int = 36,
        latent_dim: int = 20,
        num_encoder_channels: int = 30,
        num_preprocess_cells: int = 2,
        num_postprocess_cells: int = 2,
        num_encoder_cells: int = 2,
        num_decoder_cells: int = 2,
        num_output_mixtures: int = 10,
        encoder_resolution_group_offsets: FrozenSet[int] = frozenset([15, 23, 27, 31]),
        power_method_steps: int = 4,
    ):
        super().__init__()

        self.loss_config = loss_config
        self.register_buffer("steps", torch.tensor(0))
        self.register_buffer(
            "latent_group_weights",
            self._compute_latent_group_weights(num_latent_groups, encoder_resolution_group_offsets),
        )

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
            [num_latent_groups - 2 - x for x in encoder_resolution_group_offsets]
        )

        self.decoder = Decoder(
            in_channels=bottleneck_channels,
            in_resolution=image_resolution // bottleneck_scale_factor,
            num_latent_groups=num_latent_groups,
            num_cells_per_group=num_decoder_cells,
            resolution_group_offsets=decoder_resolution_group_offsets,
            latent_dim=latent_dim,
        )

        self.mid_processor = nn.Sequential(
            nn.ELU(),
            ConvBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                bias=True,
                weight_norm=True,
            ),
            nn.ELU(),
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
        self.u = {}
        self.v = {}
        grouped_weights = self._group_weights()
        for key, weight in sorted(grouped_weights.items()):
            batch, rows, cols = weight.size()
            self.u[key] = F.normalize(
                torch.randn(int(batch), int(rows), device="cuda", requires_grad=False),
                dim=1,
                eps=1e-3,
            )
            self.v[key] = F.normalize(
                torch.randn(int(batch), int(cols), device="cuda", requires_grad=False),
                dim=1,
                eps=1e-3,
            )

        print_once(f"Executing {self.power_method_steps * 10} power method steps to warm up spectral normalization...")
        start_time = time.time()
        self._power_method(grouped_weights, steps=self.power_method_steps * 10)
        print_once(f"Spectral normalization warm up took {time.time() - start_time} seconds")

    def _compute_latent_group_weights(self, num_latent_groups: int, resolution_offsets: FrozenSet[int]) -> torch.Tensor:
        sorted_offsets = [0] + sorted([num_latent_groups - 1 - x for x in resolution_offsets]) + [num_latent_groups]
        group_weights = []
        resolution = 1.0
        for start_offset, end_offset in zip(sorted_offsets[:-1], sorted_offsets[1:]):
            groups_per_resolution = end_offset - start_offset
            group_weights.append(torch.full((groups_per_resolution, ), groups_per_resolution / resolution))
            resolution *= 4.0
        weights = torch.cat(group_weights, dim=0)
        return weights / torch.min(weights)

    def batch_norm_loss(self) -> torch.Tensor:
        losses = []
        for module in self.modules():
            if isinstance(module, nn.SyncBatchNorm) and module.affine:
                losses.append(torch.max(torch.abs(module.weight)))

        return self.loss_config.reg_weight * sum(losses)

    def _group_weights(self) -> Dict[str, torch.Tensor]:
        grouped_weights = collections.defaultdict(list)
        for n, m in self.named_modules():
            # Spectral normalization is not applied to the squeeze and excitation layers.
            if isinstance(m, ConvBlock):
                weight_mat = m.weight.view(m.weight.size(0), -1)
                key = "_".join(str(d) for d in weight_mat.size())
                grouped_weights[key].append(weight_mat)

        return {key: torch.stack(sorted(weights, key=lambda t: torch.sum(t).item())) for key, weights in grouped_weights.items()}

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
        for key, weights in sorted(grouped_weights.items()):
            singular_value = torch.bmm(self.u[key].unsqueeze(dim=1), torch.bmm(weights, self.v[key].unsqueeze(dim=2)))
            losses.append(torch.sum(singular_value))

        return self.loss_config.reg_weight * sum(losses)

    def kl_loss(self, kl_divs: List[torch.Tensor]) -> torch.Tensor:
        # Create B x G matrix with the KL divergence sums for each example x latent group.
        kls = torch.stack([torch.sum(kl, dim=[1, 2, 3]) for kl in kl_divs], dim=1)
        # Compute a G-size tensor with the mean weight of each latent group with a smoothing factor of (0.01)
        # and also normalize wrt resolution and number of groups with resolution.
        kl_group_coeffs = (torch.mean(kls, dim=0) + 0.01) * self.latent_group_weights
        # Normalize the kl group coefficients so that they add up to num_latent_groups.
        kl_group_coeffs = kl_group_coeffs / torch.mean(kl_group_coeffs)

        # Multiply each group with the corresponding coefficient; sum across groups and average per example.
        total_kl_loss = torch.mean(torch.sum(kls * kl_group_coeffs.detach(), dim=1))
        # Multiply loss with its weight according to the current training schedule.
        return self.loss_config.kl_coeff(self.steps) * total_kl_loss

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        images = images.to(device="cuda")

        x = self.preprocessor(images)

        x, encoder_states = self.encoder(x)

        x = self.mid_processor(x)

        x, kl_divs, log_qs, log_ps = self.decoder(encoder_final_state=x, encoder_states=encoder_states)

        x = self.postprocessor(x)

        x = self.image_conditional(x)

        dist = LogisticMixture(num_mixtures=self.num_output_mixtures, params=x)

        log_probs = -dist.log_p(images)
        reconstruction_loss = torch.mean(torch.sum(log_probs, dim=[1, 2]))

        losses = {
            self.RECON_LOSS: reconstruction_loss,
            self.KL_LOSS: self.kl_loss(kl_divs),
            self.BN_LOSS: self.batch_norm_loss(),
            self.SPECTRAL_LOSS: self.spectral_loss(),
            self.KL_COEFF: self.loss_config.kl_coeff(self.steps),
        }
        self.steps += 1

        return {
            "loss": sum(losses[k] for k in self.LOSS_KEYS),
            **losses,
        }

    def sample(self, batch_size: int, temperature: float) -> torch.Tensor:
        x = self.decoder.sample(batch_size=batch_size)
        x = self.postprocessor(x)
        x = self.image_conditional(x)
        return LogisticMixture(num_mixtures=self.num_output_mixtures, params=x).sample(temperature=temperature)

    def summarize(self, step: int, epoch: int, raw_metrics: Dict[str, Any]) -> None:
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in raw_metrics.items()}
        formatted_metrics = "\t".join([f"{k}: {metrics[k]:6f}" for k in metrics])
        print(f"Step {step}:\t{formatted_metrics}")
