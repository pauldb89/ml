import random

import torch

from board_games.ticket2ride.features import ALL_EXTRACTORS
from board_games.ticket2ride.game_logic import Game
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.policies import UniformRandomPolicy, ModelPolicy


def test_determinism_uniform_random() -> None:
    random.seed(0)
    game = Game(policies=[UniformRandomPolicy() for _ in range(2)])
    game_stats = game.run()
    assert game_stats.total_points == [-33, 51]



def test_determinism_model() -> None:
    random.seed(0)
    torch.manual_seed(0)
    model = Model(extractors=ALL_EXTRACTORS, layers=2, dim=128, heads=4, rel_window=10)
    game = Game(policies=[ModelPolicy(model=model) for _ in range(2)])
    game_stats = game.run()
    assert game_stats.total_points == [-48, 27]
