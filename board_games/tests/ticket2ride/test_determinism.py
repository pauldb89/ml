import random

from board_games.ticket2ride.logic import Game, UniformRandomPolicy


def test_determinism() -> None:
    random.seed(0)
    game = Game(policies=[UniformRandomPolicy() for _ in range(2)])
    game_stats = game.run()
    assert game_stats.total_points == [-57, 22]
