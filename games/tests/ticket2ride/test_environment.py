import random

import torch

from board_games.ticket2ride.environment import BatchRoller
from board_games.ticket2ride.features import DYNAMIC_EXTRACTORS
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.policies import UniformRandomPolicy, ArgmaxModelPolicy
from board_games.ticket2ride.tracker import Tracker


def test_determinism_uniform_random() -> None:
    random.seed(0)
    roller = BatchRoller()
    transitions = roller.run(
        seeds=[0],
        policies=[UniformRandomPolicy(seed=0)],
        player_policy_ids=[[0, 0]],
        tracker=Tracker()
    )
    scorecard = transitions[0][-1].score.scorecard
    assert [player_score.total_points for player_score in scorecard] == [-27, -93]


def test_determinism_model() -> None:
    random.seed(0)
    torch.manual_seed(0)
    model = Model(
        device=torch.device("cpu"),
        extractors=DYNAMIC_EXTRACTORS,
        layers=2,
        dim=128,
        heads=4,
        rel_window=10,
    )
    roller = BatchRoller()
    transitions = roller.run(
        seeds=[0],
        policies=[ArgmaxModelPolicy(model=model)],
        player_policy_ids=[[0, 0]],
        tracker=Tracker()
    )
    scorecard = transitions[0][-1].score.scorecard
    assert [player_score.total_points for player_score in scorecard] == [-48, -80]
