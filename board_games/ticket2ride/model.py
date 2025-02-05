import collections
import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from board_games.ticket2ride.action_utils import PLAN_CLASSES, DRAW_CARD_CLASSES, \
    CHOOSE_TICKETS_CLASSES, BUILD_ROUTE_CLASSES
from board_games.ticket2ride.actions import ActionType, Action
from board_games.ticket2ride.features import FEATURE_REGISTRY, FeatureType, \
    Extractor, Features, BatchFeatures
from board_games.ticket2ride.state import ObservedState, Score


@dataclass
class RawSample:
    episode_id: int
    state: ObservedState
    action: Action
    score: Score


@dataclass
class Sample(RawSample):
    reward: float


class EmbeddingTable(nn.Module):
    def __init__(self, device: torch.device, extractors: list[Extractor], dim: int) -> None:
        super().__init__()

        self.extractors = extractors
        self.device = device

        self.offsets = {}
        size = 1
        for extractor in extractors:
            for feature_type in extractor.feature_types:
                if feature_type not in self.offsets:
                    self.offsets[feature_type] = size
                    size += FEATURE_REGISTRY[feature_type].cardinality

        self.embeddings = nn.Embedding(size, dim, padding_idx=0)

        self.to(device)

    def featurize(self, states: list[ObservedState]) -> BatchFeatures:
        batch_features = []
        for state in states:
            features = []
            for extractor in self.extractors:
                features.extend(extractor.extract(state))
            batch_features.append(features)

        return batch_features

    def compute_indices(self, batch_features: BatchFeatures) -> torch.Tensor:
        batch_indices = []
        for features in batch_features:
            indices = []
            for feature in features:
                indices.append(self.offsets[feature.type] + feature.value)
            batch_indices.append(torch.tensor(indices, device=self.device))

        return torch.nn.utils.rnn.pad_sequence(batch_indices, batch_first=True, padding_value=0)

    def forward(self, states: list[ObservedState]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.featurize(states)
        indices = self.compute_indices(features)
        return self.embeddings(indices), torch.ne(indices, 0)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class RelativeSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, rel_window: int, rel_proj: nn.Linear) -> None:
        super().__init__()

        self.heads = heads
        self.rel_window = rel_window
        self.rel_proj = rel_proj

        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.scalar_norm = math.sqrt(dim / heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, d = x.size()
        qkv = self.to_qkv(x).view(b, s, self.heads, -1)

        queries, keys, values = qkv.chunk(3, dim=-1)

        queries = torch.permute(queries, dims=[0, 2, 1, 3])
        keys = torch.permute(keys, dims=[0, 2, 1, 3])
        values = torch.permute(values, dims=[0, 2, 1, 3])

        offsets = (
              torch.arange(s, device=x.device).unsqueeze(dim=0)
              - torch.arange(s, device=x.device).unsqueeze(dim=1)
        ).clamp(min=-self.rel_window, max=self.rel_window) + self.rel_window
        offsets = offsets.repeat(b, self.heads, 1, 1)

        content_scores = torch.einsum("bhik,bhjk->bhij", queries, keys) / self.scalar_norm
        position_scores = torch.gather(self.rel_proj(queries), dim=-1, index=offsets)
        scores = content_scores + position_scores

        scores = scores.masked_fill(~mask.view(b, 1, 1, s), float("-inf"))
        attention = torch.softmax(scores, dim=-1)

        ret = torch.einsum("bhij,bhjk->bhik", attention, values)
        return ret.permute(dims=[0, 2, 1, 3]).reshape(b, s, d)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, rel_window: int, rel_proj: nn.Linear) -> None:
        super().__init__()

        self.attn_ln = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RelativeSelfAttention(
            dim=dim,
            heads=heads,
            rel_window=rel_window,
            rel_proj=rel_proj,
        )

        self.mlp_block = Residual(
            module=nn.Sequential(
                nn.LayerNorm(dim, eps=1e-6),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim),
            )
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask)
        return self.mlp_block(x)


class Model(nn.Module):
    def __init__(
        self,
        device: torch.device,
        extractors: list[Extractor],
        layers: int,
        dim: int,
        heads: int,
        rel_window: int,
    ) -> None:
        super().__init__()

        self.device = device
        self.extractors = extractors

        self.feature_index: dict[FeatureType, int] = {}
        self.embeddings = EmbeddingTable(device, extractors, dim)

        assert (
            dim % heads == 0
        ), f"The hidden dimension {dim} is not visible by the number of heads {heads}"
        rel_proj = nn.Linear(dim // heads, 2 * rel_window + 3)

        blocks = [
            TransformerBlock(dim=dim, heads=heads, rel_window=rel_window, rel_proj=rel_proj)
            for _ in range(layers)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.heads = nn.ModuleDict({
            ActionType.PLAN: nn.Linear(dim, len(PLAN_CLASSES)),
            ActionType.DRAW_CARD: nn.Linear(dim, len(DRAW_CARD_CLASSES)),
            ActionType.DRAW_TICKETS: nn.Linear(dim, len(CHOOSE_TICKETS_CLASSES)),
            ActionType.BUILD_ROUTE: nn.Linear(dim, len(BUILD_ROUTE_CLASSES)),
        })

        self.to(device)

    def featurize(self, state: ObservedState) -> Features:
        features = []
        for extractor in self.extractors:
            features.extend(extractor.extract(state))

        return features

    def forward(
        self,
        states: list[ObservedState],
        head: ActionType,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, seq_mask = self.embeddings(states)
        for block in self.blocks:
            x = block(x, seq_mask)

        x = self.norm(x)[:, 0, :]
        logits = self.heads[head](x)

        if mask is None:
            return logits

        return logits.masked_fill(mask.to(self.device) == 0, value=float("-inf"))

    def loss(self, samples: list[Sample]) -> torch.Tensor:
        grouped_samples = collections.defaultdict(list)
        for sample in samples:
            grouped_samples[sample.state.next_action].append(sample)

        losses = []
        weights = []
        for action_type, group in grouped_samples.items():
            states = []
            targets = []
            for sample in group:
                states.append(sample.state)
                weights.append(sample.reward)
                assert sample.action.class_id is not None
                targets.append(sample.action.class_id)

            logits = self(states, head=action_type)
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            loss_terms = F.cross_entropy(logits, targets, reduction="none")
            losses.append(loss_terms)

        return (torch.cat(losses, dim=0) * torch.tensor(weights, device=self.device)).mean()
