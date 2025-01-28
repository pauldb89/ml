import abc
import collections
import copy
import random
from dataclasses import asdict

import numpy as np
import torch
import wandb

from board_games.ticket2ride.common import timer
# from torch.cuda.amp import GradScaler

from board_games.ticket2ride.environment import Environment, Roller
from board_games.ticket2ride.model import Model, RawSample, Sample
from board_games.ticket2ride.policies import Policy, UniformRandomPolicy, \
    ArgmaxModelPolicy, StochasticModelPolicy
from board_games.ticket2ride.state import Transition


class Reward(abc.ABC):
    @abc.abstractmethod
    def apply(
        self,
        raw_samples: list[RawSample],
        final_transition: Transition,
        epoch_id: int,
    ) -> list[Sample]:
        ...


class PointsReward(Reward):
    def __init__(self, discount: float) -> None:
        super().__init__()
        self.discount = discount

    def apply(
        self,
        raw_samples: list[RawSample],
        final_transition: Transition,
        epoch_id: int,
    ) -> list[Sample]:
        del final_transition

        samples = []
        reward = 0
        for raw_sample in reversed(raw_samples):
            reward = self.discount * reward + raw_sample.score.turn_score.total_points
            samples.append(Sample(reward=reward, **asdict(raw_sample)))

        return list(reversed(samples))


class WinReward(Reward):
    def __init__(self, discount: float, reward: int) -> None:
        super().__init__()
        self.discount = discount
        self.reward = reward

    def apply(
        self,
        raw_samples: list[RawSample],
        final_transition: Transition,
        epoch_id: int,
    ) -> list[Sample]:
        last_sample = raw_samples[-1]
        if final_transition.score.winner_id == last_sample.state.player.id:
            reward = self.reward
        else:
            reward = -self.reward

        samples = [Sample(reward=reward, **asdict(last_sample))]
        for raw_sample in reversed(raw_samples[:-1]):
            reward = self.discount * reward
            samples.append(Sample(reward=reward, **asdict(raw_sample)))

        return list(reversed(samples))


class MixedReward(Reward):
    def __init__(
        self,
        initial_reward: Reward,
        final_reward: Reward,
        num_epochs: int,
    ) -> None:
        super().__init__()
        self.initial_reward = initial_reward
        self.final_reward = final_reward
        self.num_epochs = num_epochs

    def apply(
        self,
        raw_samples: list[RawSample],
        final_transition: Transition,
        epoch_id: int,
    ) -> list[Sample]:
        samples = []
        initial_reward_samples = self.initial_reward.apply(raw_samples, final_transition, epoch_id)
        final_reward_samples = self.final_reward.apply(raw_samples, final_transition, epoch_id)
        for initial_sample, final_sample in zip(initial_reward_samples, final_reward_samples):
            assert initial_sample.state == final_sample.state
            assert initial_sample.action == final_sample.action
            assert initial_sample.score == final_sample.score

            alpha = epoch_id / (self.num_epochs - 1)
            reward = initial_sample.reward * (1 - alpha) + final_sample.reward * alpha
            sample = Sample(reward=reward, **asdict(initial_sample))
            samples.append(sample)

        return samples


class PolicyGradientTrainer:
    def __init__(
        self,
        env: Environment,
        model: Model,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        batch_size: int,
        num_samples_per_epoch: int,
        evaluate_every_n_epochs: int,
        reward_fn: Reward,
    ):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples_per_epoch = num_samples_per_epoch
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.reward_fn = reward_fn

        # TODO(pauldb): Enable mixed precision training with GradScaler after switching to gpu.
        # self.scaler = GradScaler()

    # TODO(pauldb): Maybe make this list[Policy] so we can take actions differently for different
    # players (e.g. using older policies).
    def collect_episode(self, policy: Policy, epoch_id: int) -> dict[int, list[Sample]]:
        raw_samples = collections.defaultdict(list)
        # TODO(pauldb): Should we be sampling with a fixed seed?
        state = self.env.reset()

        while True:
            # TODO(pauldb): Does it make a difference if we cache the features and action index?
            action = policy.choose_action(state)
            transition = self.env.step(action)
            raw_samples[state.player.id].append(RawSample(state, action, score=transition.score))
            state = transition.state

            if state.terminal:
                break

        samples = collections.defaultdict(list)
        for player_id, player_raw_samples in raw_samples.items():
            samples[player_id] = self.reward_fn.apply(
                player_raw_samples, transition, epoch_id=epoch_id
            )

        return samples

    def collect_samples(self, epoch_id: int) -> list[Sample]:
        self.model.eval()
        policy = StochasticModelPolicy(model=self.model)

        samples = []
        for _ in range(self.num_samples_per_epoch):
            episode_samples = self.collect_episode(policy, epoch_id)
            for player_samples in episode_samples.values():
                samples.extend(player_samples)

        return samples

    def train(self, samples: list[Sample], epoch_id: int) -> None:
        num_batches = 0
        losses = []
        for start_index in range(0, len(samples), self.batch_size):
            if start_index + self.batch_size <= len(samples):
                num_batches += 1
                batch_samples = samples[start_index:start_index + self.batch_size]

                self.optimizer.zero_grad()
                loss = self.model.loss(batch_samples)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

        wandb.log(
            {"loss": np.mean(losses), "num_batches": num_batches},
            step=epoch_id,
        )

    def evaluate(self, epoch_id: int) -> None:
        model_policy = ArgmaxModelPolicy(model=self.model)
        random_policies = [UniformRandomPolicy() for _ in range(self.env.num_players - 1)]

        wins = []
        total_points = []
        for _ in range(self.num_samples_per_epoch):
            index = random.randint(0, len(random_policies) - 1)
            policies: list[Policy] = copy.deepcopy(random_policies)
            policies.insert(index, model_policy)
            roller = Roller(env=self.env, policies=policies)
            stats = roller.run()

            score = stats.transitions[-1].score
            wins.append(score.winner_id == index)
            total_points.append(score.scorecard[index].total_points)

        wandb.log(
            data={
                "vs_random.wins": np.mean(wins),
                "vs_random.total_points": np.mean(total_points),
            },
            step=epoch_id,
        )

    def execute(self) -> None:
        for epoch_id in range(self.num_epochs):
            with timer("t_collect_samples", step=epoch_id):
                samples = self.collect_samples(epoch_id)

            with timer("t_train", step=epoch_id):
                self.train(samples, epoch_id=epoch_id)

            if epoch_id % self.evaluate_every_n_epochs == 0:
                with timer("t_evaluate", step=epoch_id):
                    self.evaluate(epoch_id=epoch_id)
