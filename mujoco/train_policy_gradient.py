import os
from argparse import ArgumentParser

import gym
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam

from common.distributed import world_size
from common.samplers import set_seeds
from common.solvers.policy_gradient import PolicyGradientSolver
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from mujoco.model import DiscreteMLPPolicy


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--env_name", type=str, default="CartPole-v0", choices=["CartPole-v0"], help="Environment name")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of times to run the policy in the environment to collect rewards and train it",
    )
    parser.add_argument(
        "--num_samples_per_epoch",
        type=int,
        default=10_000,
        help="How many state transitions and rewards to execute per epoch in the environment"
    )
    parser.add_argument("--discount", type=float, default=0.95, help="Future returns discount factor")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers in policy")
    parser.add_argument("--skip_normalize", default=False, action="store_true", help="Skip normalizing returns")
    parser.add_argument(
        "--skip_reward_to_go",
        default=False,
        action="store_true",
        help="Skip the variance reduction trick that exploits causality by summing only over the trailing actions"
    )
    args = parser.parse_args()

    wandb_init(f"mujoco-{args.env_name}", dir=WANDB_DIR)
    wandb_config_update(args)

    train_env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name, render_mode="rgb_array")

    local_seed = args.seed * 2 * world_size() + local_rank
    set_seeds(local_seed)
    train_env.reset(seed=local_seed)
    eval_env.reset(seed=local_seed + world_size())

    policy = DiscreteMLPPolicy(env=train_env, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
    policy.cuda()
    policy = DistributedDataParallel(policy)

    optimizer = Adam(policy.parameters(), lr=args.lr)

    solver = PolicyGradientSolver(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        optimizer=optimizer,
        epochs=args.epochs,
        num_samples_per_epoch=args.num_samples_per_epoch,
        discount=args.discount,
        normalize_returns=not args.skip_normalize,
        reward_to_go=not args.skip_reward_to_go
    )

    solver.execute()


if __name__ == "__main__":
    main()
