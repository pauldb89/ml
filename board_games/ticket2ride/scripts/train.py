from argparse import ArgumentParser

from common.wandb import wandb_init, wandb_config_update


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    args = parser.parse_args()

    wandb_init(project="board_games")
    wandb_config_update(args)


if __name__ == "__main__":
    main()


