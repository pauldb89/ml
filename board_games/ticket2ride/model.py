import itertools
import math

import torch
from torch import nn

from board_games.ticket2ride.board_logic import Player, Board, RouteInfo
from board_games.ticket2ride.consts import EXTENDED_COLORS, ROUTES, ANY, COLORS
from board_games.ticket2ride.entities import Color, Route, Tickets, Card, DrawnTickets
from board_games.ticket2ride.features import Extractor, FeatureType, FEATURE_REGISTRY, FeatureValue, \
    FeatureContext, TaskType, BatchFeatures
from board_games.ticket2ride.policy_helpers import ActionType

CHOOSE_ACTION_CLASSES: list[ActionType] = [
    ActionType.DRAW_CARDS,
    ActionType.DRAW_TICKETS,
    ActionType.BUILD_ROUTE,
]


def generate_card_classes() -> list[Card | None]:
    return [None] + [Card(c) for c in EXTENDED_COLORS]


def generate_choose_ticket_classes() -> list[tuple[int, ...]]:
    return [x for k in range(3) for x in itertools.combinations(range(3), k+1)]


def generate_build_route_classes() -> list[tuple[Route, Color]]:
    classes = []
    for route in ROUTES:
        colors = [route.color] if route.color != ANY else COLORS
        for color in colors:
            classes.append((route, color))
    return classes


DRAW_CARD_CLASSES = generate_card_classes()
CHOOSE_TICKETS_CLASSES = generate_choose_ticket_classes()
BUILD_ROUTE_CLASSES = generate_build_route_classes()


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class RelativeSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, rel_window: int, rel_proj: nn.Linear) -> None:
        super().__init__()

        assert (
            dim % heads == 0
        ), f"The hidden dimension {dim} is not visible by the number of heads {heads}"

        self.heads = heads
        self.rel_window = rel_window
        self.rel_proj = rel_proj

        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.scalar_norm = math.sqrt(dim / heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        content_scores = torch.einsum("bhik,bhjk->bhij", queries, keys) / self.scalar_norm
        position_scores = torch.gather(self.rel_proj(queries), dim=-1, index=offsets)
        attention = torch.softmax(content_scores + position_scores, dim=-1)

        ret = torch.einsum("bhij,bhjk->bhik", attention, values)
        return ret.permute(dims=[0, 2, 1, 3]).view(b, s, d)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, rel_window: int, rel_proj: nn.Linear) -> None:
        super().__init__()

        self.attention_block = Residual(
            module=nn.Sequential(
                nn.LayerNorm(dim, eps=1e-6),
                RelativeSelfAttention(
                    dim=dim,
                    heads=heads,
                    rel_window=rel_window,
                    rel_proj=rel_proj,
                ),
            )
        )

        self.mlp_block = Residual(
            module=nn.Sequential(
                nn.LayerNorm(dim, eps=1e-6),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(self.attention_block(x))


class Model(nn.Module):
    def __init__(
        self,
        extractors: list[Extractor],
        layers: int,
        dim: int,
        heads: int,
        rel_window: int,
    ) -> None:
        super().__init__()

        self.extractors = extractors

        self.feature_index: dict[FeatureType, int] = {}
        self.embeddings = nn.ModuleList()
        for extractor in self.extractors:
            for feature_type in extractor.feature_types:
                if feature_type not in self.feature_index:
                    self.feature_index[feature_type] = len(self.feature_index)
                    feature = FEATURE_REGISTRY[feature_type]
                    self.embeddings.append(nn.Embedding(feature.cardinality, dim))

        rel_proj = nn.Linear(2 * rel_window + 3, dim)
        blocks = [
            TransformerBlock(dim=dim, heads=heads, rel_window=rel_window, rel_proj=rel_proj)
            for _ in range(layers)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.heads = nn.ModuleDict({
            "action": nn.Linear(dim, len(CHOOSE_ACTION_CLASSES)),
            "cards": nn.Linear(dim, len(DRAW_CARD_CLASSES)),
            "tickets": nn.Linear(dim, len(CHOOSE_TICKETS_CLASSES)),
            "routes": nn.Linear(dim, len(BUILD_ROUTE_CLASSES)),
        })

    def featurize(self, contexts: list[FeatureContext]) -> BatchFeatures:
        batch_features = []
        for context in contexts:
            features = []
            for extractor in self.extractors:
                features.extend(extractor.extract(context))

        return batch_features

    def forward(self, batch_features: BatchFeatures, head: str, mask: torch.Tensor) -> torch.Tensor:
        batch_seq = []
        for features in batch_features:
            seq = []
            for feature in features:
                seq.append(self.embeddings[self.feature_index[feature.type]](feature.value))
            batch_seq.append(torch.stack(seq, dim=0))

        x = torch.stack(batch_seq, dim=0)
        x = self.blocks(x)
        x = self.norm(x)[:, x, :]
        return self.heads[head](x).masked_fill(torch.tensor(mask) == 0, value=float("-inf"))

    def predict(self, features: BatchFeatures, head: str, mask: torch.Tensor) -> torch.Tensor:
        logits = self(features, head, mask)
        return torch.argmax(logits, dim=-1)

    def choose_action(
        self,
        board: Board,
        player: Player,
        valid_action_types: list[ActionType],
    ) -> ActionType:
        mask = []
        for action_type in CHOOSE_ACTION_CLASSES:
            mask.append(int(action_type in valid_action_types))

        batch_features = self.featurize(
            [FeatureContext(board=board, player=player, task_type=TaskType.CHOOSE_ACTION)]
        )
        action_index = self.predict(batch_features, head="action", mask=torch.tensor([mask])).item()

        return CHOOSE_ACTION_CLASSES[action_index]

    def draw_card(
        self,
        board: Board,
        player: Player,
        draw_options: list[Card | None],
    ) -> Card | None:
        mask = []
        for cls in DRAW_CARD_CLASSES:
            mask.append(int(cls in draw_options))

        batch_features = self.featurize(
            [FeatureContext(board=board, player=player, task_type=TaskType.DRAW_CARD)]
        )
        card_index = self.predict(batch_features, head="cards", mask=torch.tensor([mask])).item()
        return DRAW_CARD_CLASSES[card_index]

    def choose_tickets(
        self,
        board: Board,
        player: Player,
        drawn_tickets: DrawnTickets,
        is_initial_turn: bool,
    ) -> Tickets:
        mask = []
        for combo in CHOOSE_TICKETS_CLASSES:
            mask.append(1 if len(combo) >= 2 or not is_initial_turn else 0)

        batch_features = self.featurize(
            [
                FeatureContext(
                    board=board,
                    player=player,
                    task_type=TaskType.CHOOSE_TICKETS,
                    drawn_tickets=drawn_tickets,
                )
            ]
        )
        combo_index = self.predict(batch_features, head="tickets", mask=torch.tensor([mask])).item()
        combo = CHOOSE_TICKETS_CLASSES[combo_index]
        return tuple(drawn_tickets[ticket_idx] for ticket_idx in combo)

    def build_route(
        self,
        board: Board,
        player: Player,
        build_options: list[RouteInfo],
    ) -> RouteInfo:
        valid_options = set()
        for route_info in build_options:
            valid_options.add((ROUTES[route_info.route_id], route_info.color))

        mask = []
        for cls in BUILD_ROUTE_CLASSES:
            mask.append(int(cls in valid_options))

        batch_features = self.featurize(
            [FeatureContext(board=board, player=player, task_type=TaskType.BUILD_ROUTE)]
        )
        route_index = self.predict(batch_features, head="routes", mask=torch.tensor([mask])).item()
        route, color = BUILD_ROUTE_CLASSES[route_index]

        return RouteInfo(
            route_id=route.id,
            player_id=player.id,
            color=color,
            num_any_cards=max(0, route.length - player.card_counts[color]),
        )
