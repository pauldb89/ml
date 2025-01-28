import random

import torch

from board_games.ticket2ride.features import ALL_EXTRACTORS
from board_games.ticket2ride.environment import Roller, Environment
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.policies import UniformRandomPolicy, ArgmaxModelPolicy


def test_determinism_uniform_random() -> None:
    random.seed(0)
    game = Roller(
        env=Environment(num_players=2),
        policies=[UniformRandomPolicy() for _ in range(2)]
    )
    stats = game.run()
    for player_score in stats.scorecard:
        print(f"{player_score=}")
    assert [player_score.total_points for player_score in stats.scorecard] == [-53, -79]


def test_determinism_model() -> None:
    random.seed(0)
    torch.manual_seed(0)
    model = Model(extractors=ALL_EXTRACTORS, layers=2, dim=128, heads=4, rel_window=10)
    game = Roller(
        env=Environment(num_players=2),
        policies=[ArgmaxModelPolicy(model=model) for _ in range(2)]
    )
    stats = game.run()
    assert [player_score.total_points for player_score in stats.scorecard] == [-28, -73]
