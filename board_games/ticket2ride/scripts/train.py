from argparse import ArgumentParser

import torch
import wandb

from board_games.ticket2ride.environment import Environment
from board_games.ticket2ride.features import ALL_EXTRACTORS
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.solver import PolicyGradientSolver


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_players", type=int, default=2, help="Number of players")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--num_samples_per_epoch", type=int, default=100, help="Number of samples per epoch"
    )
    parser.add_argument("--dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--rel_window", type=int, default=100, help="Relative window size")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    wandb.init(project="board_games")
    wandb.config.update(args)

    model = Model(
        extractor=ALL_EXTRACTORS,
        dim_hidden=args.dim,
        layers=args.layers,
        heads=args.heads,
        rel_window=args.rel_window,
    )

    env = Environment()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    solver = PolicyGradientSolver(
        env=env,
        model=model,
        optimizer=optimizer,
        num_epochs=args.epochs,
        num_samples_per_epoch=args.num_samples_per_epoch,
        discount=args.discount,
    )


if __name__ == "__main__":
    main()


