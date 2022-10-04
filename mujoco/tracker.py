import random
from argparse import ArgumentParser

import gym
from torch import package
from torch.optim import Adam

from common.samplers import set_seeds
from common.solvers.behavior_cloning import BehaviorCloningSolver
from common.solvers.replay_buffer import BasicReplayBuffer
from common.wandb import WANDB_DIR
from common.wandb import wandb_config_update
from common.wandb import wandb_init
from mujoco.model import ExpertPolicyWrapper


def main():
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

    policy = ExpertPolicyWrapper(expert_policy)

    optimizer = Adam(policy.parameters(), lr=1.0)

    solver = BehaviorCloningSolver(
        train_env=train_env,
        eval_env=eval_env,
        policy=policy,
        optimizer=optimizer,
        expert_policy=expert_policy,
        replay_buffer=BasicReplayBuffer(buffer_max_size=1000),
        epochs=0,
        samples_per_epoch=0,
        steps_per_epoch=0,
        batch_size=1,
        evaluate_at_start=True,
    )

    solver.execute()


if __name__ == "__main__":
    main()
