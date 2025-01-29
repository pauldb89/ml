import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb

from board_games.ticket2ride.environment import Environment
from board_games.ticket2ride.features import ALL_EXTRACTORS
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.trainer import PolicyGradientTrainer, PointsReward


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_players", type=int, default=2, help="Number of players")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_samples_per_epoch", type=int, default=32, help="Number of samples per epoch"
    )
    parser.add_argument(
        "--evaluate_every_n_epochs", type=int, default=5, help="Evaluate every n epochs"
    )
    parser.add_argument("--dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--rel_window", type=int, default=100, help="Relative window size")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb.init(project="board_games")
    wandb.config.update(args)

    model = Model(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        extractors=ALL_EXTRACTORS,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        rel_window=args.rel_window,
    )

    env = Environment(num_players=args.num_players)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO(pauldb): Add win reward and mix with points reward. Keep scales in mind.
    trainer = PolicyGradientTrainer(
        env=env,
        model=model,
        optimizer=optimizer,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples_per_epoch=args.num_samples_per_epoch,
        evaluate_every_n_epochs=args.evaluate_every_n_epochs,
        reward_fn=PointsReward(discount=args.discount),
    )
    trainer.execute()


if __name__ == "__main__":
    main()


