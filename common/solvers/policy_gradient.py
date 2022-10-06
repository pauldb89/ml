import itertools
import time
from typing import List
from typing import NamedTuple

import gym
import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler
from torch.distributed import all_gather_object
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from common.distributed import is_root_process
from common.distributed import print_once
from common.distributed import world_size
from common.wandb import wandb_log


class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminal_states: torch.Tensor
    returns: torch.Tensor


class PolicySolver:
    def __init__(
        self,
        train_env: gym.Env,
        eval_env: gym.Env,
        policy: DistributedDataParallel,
        epochs: int,
        num_samples_per_epoch: int,
        discount: float,
        reward_to_go: bool = True,
        evaluate_every_n_epochs: int = 20,
        num_eval_episodes: int = 25,
        max_videos_to_render: int = 1,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.policy = policy

        self.epochs = epochs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.discount = discount
        self.reward_to_go = reward_to_go

        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.num_eval_episodes = num_eval_episodes
        self.max_videos_to_render = max_videos_to_render

    def compute_discounted_returns(self, rewards: List[float], terminal_states: List[bool]) -> torch.Tensor:
        if self.reward_to_go:
            cumulative_reward = 0
            cumulative_rewards = []
            for reward, terminal_state in reversed(list(zip(rewards, terminal_states))):
                future_reward = 0 if terminal_state else self.discount * cumulative_reward
                cumulative_reward = reward + future_reward
                cumulative_rewards.append(cumulative_reward)

            return torch.tensor(list(reversed(cumulative_rewards)))

        cumulative_discount = 1
        cumulative_reward = 0
        cumulative_rewards = []
        for reward, terminal_state in zip(rewards, terminal_states):
            cumulative_reward += cumulative_discount * reward
            cumulative_rewards.append(cumulative_reward)
            cumulative_discount *= self.discount
            if terminal_state:
                cumulative_discount = 1
                cumulative_reward = 0

        for idx in reversed(range(len(cumulative_rewards) - 1)):
            if not terminal_states[idx]:
                cumulative_rewards[idx] = cumulative_rewards[idx + 1]

        return torch.tensor(cumulative_rewards)

    def collect_data(self, epoch: int) -> Batch:
        start_time = time.time()
        self.policy.eval()

        states, actions, rewards, terminal_states = [], [], [], []

        # TODO(pauldb): Either add parallelization or batching.
        episode_rewards = []
        state, _ = self.train_env.reset()
        total_reward = 0
        for _ in range(self.num_samples_per_epoch):
            with torch.inference_mode():
                action = self.policy.module.sample(torch.from_numpy(state)).cpu().item()

            next_state, reward, terminal_state, truncated, _ = self.train_env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminal_states.append(terminal_state or truncated)
            total_reward += reward

            if terminal_state or truncated:
                state = self.train_env.reset()[0]
                episode_rewards.append(total_reward)
                total_reward = 0
            else:
                state = next_state

        self.policy.train()

        metrics = {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }
        wandb_log(metrics, step=epoch)
        print_once(
            f"Epoch: {epoch} Avg Reward: {metrics['avg_reward']:.2f} +/- {metrics['std_reward']:.2f} "
            f"Max Reward: {metrics['max_reward']:.2f} Simulation time: {time.time() - start_time:.2f} seconds"
        )

        return Batch(
            states=torch.from_numpy(np.stack(states, axis=0)),
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            terminal_states=torch.tensor(terminal_states),
            returns=self.compute_discounted_returns(rewards, terminal_states)
        )

    def train_step(self, epoch: int, batch: Batch) -> None:
        pass

    def evaluate(self, epoch: int, should_render: bool) -> None:
        self.policy.eval()

        rewards = []
        videos = []
        for episode_idx in range(self.num_eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            terminal_state = False
            frames = []

            while not terminal_state:
                with torch.inference_mode():
                    action = self.policy.module.sample(torch.from_numpy(state)).cpu().item()

                next_state, reward, terminal_state, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                terminal_state |= truncated
                state = next_state

                if should_render and episode_idx < self.max_videos_to_render and is_root_process():
                    frames.append(self.eval_env.render())

            rewards.append(episode_reward)
            if frames:
                frames = np.stack(frames, axis=0)
                videos.append(np.transpose(frames, axes=[0, 3, 1, 2]))

        all_rewards = [[] for _ in range(world_size())]
        all_gather_object(all_rewards, rewards)

        if is_root_process():
            all_rewards = list(itertools.chain(*all_rewards))
            metrics = {
                "eval_avg_reward": np.mean(all_rewards),
                "eval_std_reward": np.std(all_rewards),
                "eval_max_reward": np.max(all_rewards),
            }
            wandb_log(metrics, step=epoch)

            for idx, video in enumerate(videos):
                wandb_log({f"eval_video_{idx}": wandb.Video(video, fps=15, format="gif")}, step=epoch)

            print_once(
                f"Epoch: {epoch} "
                f"Eval Avg Reward: {metrics['eval_avg_reward']:.2f} +/- {metrics['eval_std_reward']:.2f} "
                f"Eval Max Reward: {metrics['eval_max_reward']:.2f}"
            )

        self.policy.train()

    def execute(self) -> None:
        for epoch in range(self.epochs):
            batch = self.collect_data(epoch=epoch)

            self.train_step(epoch=epoch, batch=batch)

            next_epoch = epoch + 1
            if next_epoch % self.evaluate_every_n_epochs == 0:
                self.evaluate(epoch=next_epoch, should_render=True)


class PolicyGradientSolver(PolicySolver):
    def __init__(
        self,
        train_env: gym.Env,
        eval_env: gym.Env,
        policy: DistributedDataParallel,
        optimizer: Optimizer,
        epochs: int,
        num_samples_per_epoch: int,
        discount: float,
        normalize_returns: bool,
        reward_to_go: bool,
    ):
        super().__init__(
            train_env=train_env,
            eval_env=eval_env,
            policy=policy,
            epochs=epochs,
            num_samples_per_epoch=num_samples_per_epoch,
            discount=discount,
            reward_to_go=reward_to_go,
        )

        self.optimizer = optimizer
        self.normalize_returns = normalize_returns

        self.scaler = GradScaler()

    def train_step(self, epoch: int, batch: Batch) -> None:
        if self.normalize_returns:
            # TODO(pauldb): I think here it's actually recommended to normalize at a single timestep t
            # across all trajectories (as opposed to all returns).
            returns = (batch.returns - torch.mean(batch.returns)) / (torch.std(batch.returns) + 1e-5)
        else:
            returns = batch.returns

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss_metrics = self.policy.module.policy_gradient_loss(
                states=batch.states,
                actions=batch.actions,
                returns=returns,
            )
        loss = loss_metrics["loss"]
        loss = self.scaler.scale(loss)
        loss.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
