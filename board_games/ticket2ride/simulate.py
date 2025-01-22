import random
import time
from argparse import ArgumentParser

import torch

from board_games.ticket2ride.features import ALL_EXTRACTORS
from board_games.ticket2ride.game_logic import Game
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.policies import KeyboardInputPolicy, ModelPolicy


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--manual', action='store_true', default=False)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    model = Model(extractors=ALL_EXTRACTORS, layers=2, dim=128, heads=4, rel_window=10)

    if args.manual:
        policies = [KeyboardInputPolicy(), ModelPolicy(model)]
    else:
        policies = [ModelPolicy(model=model) for _ in range(2)]

    game = Game(policies=policies)

    # game = Game(policies=[KeyboardInputPolicy(), UniformRandomPolicy()])
    start_time = time.time()
    game_stats = game.run(verbose=True)
    print(f"Running the game took {time.time() - start_time} seconds")
    print(game_stats)


if __name__ == "__main__":
    main()
