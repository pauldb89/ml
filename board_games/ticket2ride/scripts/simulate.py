import random
import time
from argparse import ArgumentParser

import torch

from board_games.ticket2ride.features import ALL_EXTRACTORS
from board_games.ticket2ride.environment import Roller, Environment
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.policies import KeyboardInputPolicy, ArgmaxModelPolicy, UniformRandomPolicy
from board_games.ticket2ride.render_utils import print_scorecard, print_board


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--manual', action='store_true', default=False)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if args.manual:
        policies = [KeyboardInputPolicy(), UniformRandomPolicy(seed=0)]
    else:
        model = Model(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            extractors=ALL_EXTRACTORS,
            layers=2,
            dim=128,
            heads=4,
            rel_window=10,
        )
        policies = [ArgmaxModelPolicy(model=model) for _ in range(2)]

    game = Roller(env=Environment(num_players=2), policies=policies)

    start_time = time.time()
    game_stats = game.run(verbose=True, seed=0)
    print(f"Running the game took {time.time() - start_time} seconds")
    print_board(game_stats.transitions[-1].state.board)
    print_scorecard(game_stats.scorecard)


if __name__ == "__main__":
    main()
