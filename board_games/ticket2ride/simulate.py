import random

from board_games.ticket2ride.game_logic import Game
from board_games.ticket2ride.policies import KeyboardInputPolicy, UniformRandomPolicy


def main() -> None:
    random.seed(0)
    policies = [KeyboardInputPolicy(), UniformRandomPolicy()]
    game = Game(policies=policies)
    game_stats = game.run(verbose=True)
    print(game_stats)


if __name__ == "__main__":
    main()
