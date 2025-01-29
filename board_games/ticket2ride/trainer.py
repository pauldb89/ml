import abc
import collections
import copy
import random

import neptune
import torch
import tqdm

from torch.cuda.amp import GradScaler

from board_games.ticket2ride.board import InvalidGameStateError
from board_games.ticket2ride.environment import Environment, Roller
from board_games.ticket2ride.model import Model, RawSample, Sample
from board_games.ticket2ride.policies import Policy, UniformRandomPolicy, \
    ArgmaxModelPolicy, StochasticModelPolicy
from board_games.ticket2ride.state import Transition
from board_games.ticket2ride.tracker import Tracker


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
            samples.append(
                Sample(
                    state=raw_sample.state,
                    action=raw_sample.action,
                    score=raw_sample.score,
                    reward=reward,
                )
            )

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

        samples = [
            Sample(
                state=last_sample.state,
                action=last_sample.action,
                score=last_sample.score,
                reward=reward,
            )
        ]
        for raw_sample in reversed(raw_samples[:-1]):
            reward = self.discount * reward
            samples.append(
                Sample(
                    state=raw_sample.state,
                    action=raw_sample.action,
                    score=raw_sample.score,
                    reward=reward,
                )
            )

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
            sample = Sample(
                state=initial_sample.state,
                action=initial_sample.action,
                score=initial_sample.score,
                reward=reward,
            )
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

        self.scaler = GradScaler()

    # TODO(pauldb): Maybe make this list[Policy] so we can take actions differently for different
    # players (e.g. using older policies).
    def collect_episode(
        self,
        policy: Policy,
        tracker: Tracker,
        epoch_id: int,
    ) -> dict[int, list[Sample]]:
        raw_samples = collections.defaultdict(list)

        # TODO(pauldb): Should we be sampling with a fixed seed?
        state = self.env.reset()

        while True:
            with tracker.timer("t_policy_choose_action"):
                action = policy.choose_action(state)

            with tracker.timer("t_env_step"):
                transition = self.env.step(action)

            raw_samples[state.player.id].append(RawSample(state, action, score=transition.score))
            state = transition.state

            if state.terminal:
                break

        samples = collections.defaultdict(list)
        with tracker.timer("t_compute_rewards"):
            for player_id, player_raw_samples in raw_samples.items():
                samples[player_id] = self.reward_fn.apply(
                    player_raw_samples, transition, epoch_id=epoch_id
                )

        return samples

    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()
        policy = StochasticModelPolicy(model=self.model)

        samples = []
        num_invalid_episodes = 0
        for _ in tqdm.tqdm(range(self.num_samples_per_epoch)):
            with tracker.timer("t_episode"):
                try:
                    episode_samples = self.collect_episode(policy, tracker, epoch_id)
                    for player_samples in episode_samples.values():
                        samples.extend(player_samples)
                except InvalidGameStateError:
                    num_invalid_episodes += 1

        tracker.log_value("invalid_episodes", num_invalid_episodes)

        return samples

    def train(self, samples: list[Sample], tracker: Tracker) -> None:
        self.model.train()

        num_batches = 0
        for start_index in range(0, len(samples), self.batch_size):
            if start_index + self.batch_size <= len(samples):
                num_batches += 1
                batch_samples = samples[start_index:start_index + self.batch_size]

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self.model.loss(batch_samples)

                assert torch.isnan(loss).sum() == 0

                tracker.log_value("loss", loss.item())

                loss = self.scaler.scale(loss)
                loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            tracker.log_value("num_batches", num_batches)

    def evaluate(self, tracker: Tracker) -> None:
        self.model.eval()

        model_policy = ArgmaxModelPolicy(model=self.model)
        random_policies = [UniformRandomPolicy() for _ in range(self.env.num_players - 1)]

        with tracker.scope("vs_random"):
            for _ in range(self.num_samples_per_epoch):
                index = random.randint(0, len(random_policies) - 1)
                policies: list[Policy] = copy.deepcopy(random_policies)
                policies.insert(index, model_policy)
                roller = Roller(env=self.env, policies=policies)
                stats = roller.run()

                score = stats.transitions[-1].score
                tracker.log_value("total_points", score.scorecard[index].total_points)
                tracker.log_value("wins", score.winner_id == index)

    def execute(self, run: neptune.Run) -> None:
        for epoch_id in range(self.num_epochs):
            tracker = Tracker()
            with tracker.timer("t_overall"):
                with tracker.scope("collect_samples"):
                    with tracker.timer("t_overall"):
                        samples = self.collect_samples(tracker=tracker, epoch_id=epoch_id)

                with tracker.scope("train"):
                    with tracker.timer("t_overall"):
                        self.train(samples=samples, tracker=tracker)

                if epoch_id % self.evaluate_every_n_epochs == 0:
                    with tracker.scope("eval"):
                        with tracker.timer("t_overall"):
                            self.evaluate(tracker)

            metrics = tracker.report()
            for key, value in metrics.items():
                run[key].append(value, step=epoch_id)

            print(metrics["collect_samples/invalid_episodes_sum"])
            print(
                f"Epoch {epoch_id}: Loss {metrics['train/loss_mean']}, "
                f"Total time {metrics['t_overall_mean']}"
            )
