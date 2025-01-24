from argparse import ArgumentParser

import wandb


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    args = parser.parse_args()

    wandb.init(project="board_games")
    wandb.config.update(args)


if __name__ == "__main__":
    main()


