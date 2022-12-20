import os
from argparse import ArgumentParser

import torch.distributed

from common.distributed import world_size
from common.samplers import set_seeds
from wordle.discount_schedule import ConstantDiscountSchedule
from wordle.environment import WordleEnv
from wordle.model import TransformerPolicy
from wordle.solver import WordleSolver
from wordle.vocabulary import vocabulary


def main():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    set_seeds(local_rank)

    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000, help="Number of iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--episodes_per_iteration", type=int, default=1000, help="Number of episodes per iteration")
    parser.add_argument("--steps_per_iteration", type=int, default=100, help="Number of training steps per iteration")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--discount_gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--non_existent_reward", type=float, default=0, help="Reward for a non existing letter")
    parser.add_argument("--matching_reward", type=float, default=1.0, help="Reward for a letter that exists")
    parser.add_argument("--exact_match_reward", type=float, default=3.0, help="Reward for an exact match")
    parser.add_argument(
        "--reshape_reward_max_iteration",
        type=int,
        default=None,
        help="For how many iterations to apply reward shaping. By default apply to all.",
    )
    args = parser.parse_args()

    policy = TransformerPolicy(vocabulary=vocabulary)
    policy.cuda()

    train_env = WordleEnv()
    train_env.reset(seed=local_rank)
    eval_env = WordleEnv()
    eval_env.reset(seed=world_size() + local_rank)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    solver = WordleSolver(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        optimizer=optimizer,
        discount_schedule=ConstantDiscountSchedule(gamma=args.discount_gamma),
        rewards={
            "0": args.non_existent_reward,
            "1": args.matching_reward,
            "2": args.exact_match_reward,
        },
        reshape_rewards_max_iteration=args.reshape_rewards_max_iteration,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        steps_per_iteration=args.steps_per_iteration,
        batch_size=args.batch_size,
        evaluate_every_n_iterations=100,
    )
    solver.execute()


if __name__ == "__main__":
    main()
