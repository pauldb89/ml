import os
from argparse import ArgumentParser

import gym
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam

from common.distributed import world_size
from common.samplers import set_seeds
from common.solvers.actor_critic import ActorCriticSolver
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from mujoco.model import BaselineEstimator
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
    parser.add_argument(
        "--baseline_train_steps_per_epoch",
        type=int,
        default=100,
        help="Number of optimization iterations to perform to optimize the baseline estimator per epoch",
    )
    parser.add_argument(
        "--baseline_batch_size",
        type=int,
        default=100,
        help="Batch size for updating the baseline estimator"
    )
    parser.add_argument(
        "--normalize_advantage",
        default=False,
        action="store_true",
        help="Whether to normalize the advantages to have mean 0 and std 1",
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

    baseline_estimator = BaselineEstimator(env=train_env, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
    baseline_estimator.cuda()
    baseline_estimator = DistributedDataParallel(baseline_estimator)

    policy_optimizer = Adam(policy.parameters(), lr=args.lr)
    baseline_optimizer = Adam(baseline_estimator.parameters(), lr=args.lr)

    solver = ActorCriticSolver(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        policy_optimizer=policy_optimizer,
        baseline_estimator=baseline_estimator,
        baseline_optimizer=baseline_optimizer,
        epochs=args.epochs,
        num_samples_per_epoch=args.num_samples_per_epoch,
        discount=args.discount,
        baseline_train_steps_per_epoch=args.baseline_train_steps_per_epoch,
        baseline_batch_size=args.baseline_batch_size,
        normalize_advantage=args.normalize_advantage,
    )

    solver.execute()


if __name__ == "__main__":
    main()
