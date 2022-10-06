import collections
import time
from typing import Dict
from typing import Protocol

import gym
import numpy as np
import torch
import tqdm
import wandb
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from common.solvers.replay_buffer import BasicReplayBuffer
from common.solvers.replay_buffer import ReplayBatch
from common.wandb import wandb_log


class Policy(Protocol):
    def imitation_loss(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        ...


class BehaviorCloningSolver:
    def __init__(
        self,
        train_env: gym.Env,
        eval_env: gym.Env,
        policy: Policy,
        optimizer: Optimizer,
        expert_policy: nn.Module,
        replay_buffer: BasicReplayBuffer,
        epochs: int,
        samples_per_epoch: int,
        steps_per_epoch: int,
        batch_size: int,
        evaluate_at_start: bool = False,
        evaluate_every_n_epochs: int = 1,
        num_eval_episodes: int = 25,
        render_every_n_epochs: int = 1,
        max_videos_to_log: int = 1,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.policy = policy
        self.optimizer = optimizer

        assert not expert_policy.training
        self.expert_policy = expert_policy
        self.replay_buffer = replay_buffer

        self.epochs = epochs
        self.samples_per_epoch = samples_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.evaluate_at_start = evaluate_at_start
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.num_eval_episodes = num_eval_episodes
        self.render_every_n_epochs = render_every_n_epochs
        self.max_videos_to_log = max_videos_to_log

        self.scaler = GradScaler()

    def collect_data(self) -> None:
        state, _ = self.train_env.reset()
        states, actions, rewards, terminal_states, next_states = [], [], [], [], []
        for _ in range(self.samples_per_epoch):
            states.append(state)
            action = np.squeeze(self.expert_policy.get_action(state), axis=0)
            actions.append(action)

            next_state, reward, terminal_state, truncated, _ = self.train_env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminal_states.append(terminal_state)

            state = self.train_env.reset()[0] if terminal_state or truncated else next_state

        self.replay_buffer.add(
            ReplayBatch(
                states=torch.from_numpy(np.stack(states, axis=0)).to(torch.float),
                next_states=torch.from_numpy(np.stack(next_states, axis=0)).to(torch.float),
                actions=torch.from_numpy(np.stack(actions, axis=0)),
                rewards=torch.tensor(rewards, dtype=torch.float),
                terminal_states=torch.tensor(terminal_states),
            )
        )

    def train_step(self, epoch: int) -> None:
        loss_history = collections.deque(maxlen=100)
        for _ in range(self.steps_per_epoch):
            batch = self.replay_buffer.sample(batch_size=self.batch_size)

            with torch.cuda.amp.autocast():
                loss_metrics = self.policy.imitation_loss(batch.states, batch.actions)
                total_loss = loss_metrics["loss"]
                loss_history.append(total_loss.item())

                self.optimizer.zero_grad()

                total_loss = self.scaler.scale(total_loss)
                total_loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        print(f"Epoch {epoch}: Loss {np.mean(loss_history)}")
        wandb_log({"loss": np.mean(loss_history)}, step=epoch)

    def evaluate(self, epoch: int) -> None:
        start_time = time.time()
        print(f"Running evaluation at epoch {epoch}")
        rewards = []
        videos = []
        for episode_id in tqdm.tqdm(range(self.num_eval_episodes)):
            state, _ = self.eval_env.reset()
            episodic_reward = 0
            terminal_state = False
            truncated = False
            frames = []
            episode_steps = 0

            while not (terminal_state or truncated):
                with torch.inference_mode():
                    action = self.policy.sample(torch.from_numpy(state).to(torch.float)).cpu().numpy()
                next_state, reward, terminal_state, truncated, _ = self.eval_env.step(action)
                episode_steps += 1

                if epoch % self.render_every_n_epochs == 0 and episode_id < self.max_videos_to_log:
                    frames.append(self.eval_env.render())

                episodic_reward += reward
                state = next_state

            rewards.append(episodic_reward)
            if frames:
                videos.append(np.transpose(np.stack(frames, axis=0), axes=[0, 3, 1, 2]))

        print(f"Evaluation took {time.time() - start_time} seconds...")

        metrics = {
            "eval_avg_reward": np.mean(rewards),
            "eval_max_reward": np.max(rewards),
            "eval_std_reward": np.std(rewards),
        }
        print(
            f"Epoch {epoch}: "
            f"Avg Reward {metrics['eval_avg_reward']} +/- {metrics['eval_std_reward']} "
            f"Max Reward {metrics['eval_max_reward']}"
        )
        wandb_log(metrics, step=epoch)
        for idx in range(len(videos)):
            wandb_log({f"eval_video_{idx}": wandb.Video(videos[idx], fps=15, format="gif")}, step=epoch)

    def execute(self) -> None:
        if self.evaluate_at_start:
            self.evaluate(epoch=0)

        for epoch in range(self.epochs):
            self.collect_data()
            self.train_step(epoch=epoch)

            if epoch % self.evaluate_every_n_epochs == 0:
                self.evaluate(epoch=epoch+1)
