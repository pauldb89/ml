import random
from argparse import ArgumentParser

import gym
import torch.distributed
from torch import package
from torch.optim import Adam

from common.samplers import set_seeds
from common.solvers.behavior_cloning import BehaviorCloningSolver
from common.solvers.replay_buffer import BasicReplayBuffer
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from mujoco.model import GaussianMLPPolicy


def main():
    torch.cuda.set_device(0)

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed to initialize models and environments")
    parser.add_argument(
        "--env_name",
        type=str,
        default="HalfCheetah-v2",
        choices=["Ant-v2", "HalfCheetah-v2"],
        help="OpenAI Gym environment",
    )
    parser.add_argument(
        "--expert_policy",
        type=str,
        default="/models/berkeley/expert_policies/HalfCheetah",
        help="Expert policy for annotating data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs: "
             "1 is equivalent with typical supervised learning (minus iid assumptions) "
             "> 1 is equivalent to dagger"
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=1_000,
        help="How many new samples to collect every epoch from the expert policy in Dagger",
    )
    parser.add_argument(
        "--buffer_max_size",
        type=int,
        default=1_000_000,
        help="Maximum replay buffer size",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1_000,
        help="How many training steps to perform per epoch",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Size of hidden layer")
    args = parser.parse_args()

    set_seeds(args.seed)

    wandb_init(project=f"mujoco-{args.env_name}", dir=WANDB_DIR)
    wandb_config_update(args)

    train_env = gym.make(args.env_name, render_mode=None)
    eval_env = gym.make(args.env_name, render_mode="rgb_array")
    train_env.reset(seed=random.randint(0, 1000))
    eval_env.reset(seed=random.randint(0, 1000))

    importer = package.PackageImporter(args.expert_policy)
    expert_policy = importer.load_pickle("expert", "model.pt")
    expert_policy.eval()

    policy = GaussianMLPPolicy(env=train_env, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
    policy.cuda()

    optimizer = Adam(policy.parameters(), lr=args.lr)

    solver = BehaviorCloningSolver(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        optimizer=optimizer,
        expert_policy=expert_policy,
        replay_buffer=BasicReplayBuffer(buffer_max_size=args.buffer_max_size),
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )

    solver.execute()


if __name__ == "__main__":
    main()
