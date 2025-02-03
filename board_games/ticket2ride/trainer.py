import abc
import collections
import copy
import json
import random

import neptune
import numpy as np
import torch
import tqdm

from torch.cuda.amp import GradScaler

from board_games.ticket2ride.actions import ActionType, Action, DrawCard
from board_games.ticket2ride.color import ANY
from board_games.ticket2ride.environment import Environment, Roller
from board_games.ticket2ride.model import Model, RawSample, Sample
from board_games.ticket2ride.policies import Policy, UniformRandomPolicy, \
    ArgmaxModelPolicy, StochasticModelPolicy
from board_games.ticket2ride.state import Transition
from board_games.ticket2ride.tracker import Tracker


# TODO(pauldb): Make rewards zero sum subtracting opponent average reward?
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
    def __init__(
        self,
        discount: float,
        initial_draw_card_reward: float,
        final_draw_card_reward: float,
        draw_card_horizon_epochs: int,
    ) -> None:
        super().__init__()
        self.discount = discount
        self.initial_draw_card_reward = initial_draw_card_reward
        self.final_draw_card_reward = final_draw_card_reward
        self.draw_card_horizon_epochs = draw_card_horizon_epochs

    def apply(
        self,
        raw_samples: list[RawSample],
        final_transition: Transition,
        epoch_id: int,
    ) -> list[Sample]:
        del final_transition

        samples = []
        reward = 0
        rate = min(epoch_id / self.draw_card_horizon_epochs, 1)
        draw_card_bonus = (
            (1 - rate) * self.initial_draw_card_reward + rate * self.final_draw_card_reward
        )
        for raw_sample in reversed(raw_samples):
            current_reward = raw_sample.score.turn_score.total_points
            if raw_sample.action.action_type == ActionType.DRAW_CARD:
                current_reward += draw_card_bonus

            reward = self.discount * reward + current_reward
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


def get_action_stats(actions: list[Action]) -> dict[str, float]:
    new_turn = True
    consecutive_draws = 0
    longest_draw = 0
    draw_lengths = collections.defaultdict(int)
    deck_card_draws = 0
    color_card_draws = 0
    any_card_draws = 0
    for action in actions:
        if isinstance(action, DrawCard):
            if action.card is None:
                deck_card_draws += 1
            elif action.card.color == ANY:
                any_card_draws += 1
            else:
                color_card_draws += 1

        if new_turn:
            if action.action_type == ActionType.DRAW_CARD:
                consecutive_draws += 1
            else:
                if consecutive_draws:
                    draw_lengths[consecutive_draws] += 1
                consecutive_draws = 0

        longest_draw = max(longest_draw, consecutive_draws)
        new_turn = action.action_type == ActionType.PLAN

    metrics = {
        "longest_draw": longest_draw,
        "card_draws_any": any_card_draws,
        "card_draws_deck": deck_card_draws,
        "card_draws_color": color_card_draws,
    }
    for length, cnt in draw_lengths.items():
        metrics[f"consecutive_draws_len_{length:02d}"] = cnt
    return metrics


def normalize_rewards(samples: list[Sample]) -> list[Sample]:
    rewards = [s.reward for s in samples]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    for s in samples:
        s.reward = (s.reward - mean_reward) / (std_reward + 1e-6)
    return samples


class PolicyGradientTrainer:
    def __init__(
        self,
        env: Environment,
        model: Model,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_samples_per_epoch: int,
        num_eval_samples_per_epoch: int,
        batch_size: int,
        evaluate_every_n_epochs: int,
        reward_fn: Reward,
    ):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.num_eval_samples_per_epoch = num_eval_samples_per_epoch
        self.batch_size = batch_size
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.reward_fn = reward_fn

        self.scaler = GradScaler()

        self.episode_id = 0

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
        self.episode_id += 1
        state = self.env.reset(self.episode_id)

        while True:
            with tracker.timer("t_policy_choose_action"):
                action = policy.choose_action(state)

            with tracker.timer("t_env_step"):
                transition = self.env.step(action)

            raw_samples[state.player.id].append(RawSample(state, action, score=transition.score))
            state = transition.state

            if state.terminal:
                break

        # print("-" * 20)

        samples = collections.defaultdict(list)
        with tracker.timer("t_compute_rewards"):
            for player_id, player_raw_samples in raw_samples.items():
                samples[player_id] = self.reward_fn.apply(
                    player_raw_samples, transition, epoch_id=epoch_id
                )

        # for s in samples[0]:
        #     print(s.state.turn_id, s.state.player.id, s.action, s.reward)
        # print("-" * 20)

        return samples

    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()
        policy = StochasticModelPolicy(model=self.model)

        samples = []
        for _ in tqdm.tqdm(range(self.num_samples_per_epoch), desc="Sample collection"):
            with tracker.timer("t_episode"):
                episode_samples = self.collect_episode(policy, tracker, epoch_id)
                action_rewards = collections.defaultdict(list)
                for player_samples in episode_samples.values():
                    # print([(s.action.action_type.value, s.reward) for s in player_samples])
                    for s in player_samples:
                        action_rewards[s.action.action_type.value].append(s.reward)
                    # for action_type, rewards in sorted(action_rewards.items()):
                    #     print(f"{action_type} {np.mean(rewards)}")
                    samples.extend(player_samples)

        # action_counts = collections.defaultdict(int)
        plan_rewards = collections.defaultdict(list)
        for s in samples:
            # action_counts[(s.action.action_type, s.action.class_id)] += 1
            if s.action.action_type == ActionType.PLAN:
                plan_rewards[s.action.class_id].append(s.reward)

        for class_id, rewards in sorted(plan_rewards.items()):
            print(f"{class_id=} {np.mean(rewards)=}")

        return normalize_rewards(samples)

    def train(self, samples: list[Sample], tracker: Tracker) -> None:
        self.model.train()

        self.optimizer.zero_grad()

        num_batches = 0
        total_loss = 0
        for start_index in tqdm.tqdm(range(0, len(samples), self.batch_size), desc="Train step"):
            num_batches += 1
            batch_samples = samples[start_index:start_index + self.batch_size]

            with torch.cuda.amp.autocast():
                loss = len(batch_samples) / len(samples) * self.model.loss(batch_samples)
                assert torch.isnan(loss).sum() == 0
                total_loss += loss.item()
                loss = self.scaler.scale(loss)
                loss.backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        tracker.log_value("loss", total_loss)
        tracker.log_value("num_batches", num_batches)

    def evaluate(self, tracker: Tracker, epoch_id: int) -> None:
        self.model.eval()

        model_policy = ArgmaxModelPolicy(model=self.model)
        random_policies = [UniformRandomPolicy(seed=0) for _ in range(self.env.num_players - 1)]

        with tracker.scope("vs_random"):
            for idx in tqdm.tqdm(range(self.num_eval_samples_per_epoch), desc="Eval step"):
                index = random.randint(0, len(random_policies))
                policies: list[Policy] = copy.deepcopy(random_policies)
                policies.insert(index, model_policy)
                roller = Roller(env=self.env, policies=policies)
                stats = roller.run(seed=idx)

                actions = []
                for t in stats.transitions:
                    if t.action.player_id == index:
                        actions.append(t.action)

                # for action in actions:
                #     print(action)
                # print("-" * 20)

                action_metrics = get_action_stats(actions)
                for key, value in action_metrics.items():
                    tracker.log_value(key, value)

                score = stats.transitions[-1].score
                policy_score = score.scorecard[index]
                tracker.log_value("wins", score.winner_id == index)
                tracker.log_value("total_points", policy_score.total_points)
                tracker.log_value("route_points", policy_score.route_points)
                tracker.log_value("tickets_drawn", policy_score.total_tickets)
                tracker.log_value("tickets_completed", policy_score.completed_tickets)
                tracker.log_value("ticket_points", policy_score.ticket_points)
                tracker.log_value("longest_path_bonus", policy_score.longest_path_bonus)
                tracker.log_value("completed_routes", sum(policy_score.owned_routes_by_length.values()))
                for length in range(1, 7):
                    num_routes = policy_score.owned_routes_by_length[length]
                    tracker.log_value(f"completed_routes_len_{length}", num_routes)
                    for _ in range(num_routes):
                        tracker.log_value(f"completed_routes_length", length)

        eval_metrics = {}
        for key, value in tracker.report().items():
            if key.startswith("eval/vs_random") and key.endswith("mean"):
                eval_metrics[key] = value

        print(f"Eval metrics at epoch {epoch_id}: {json.dumps(eval_metrics, indent=2, sort_keys=True)}")

    def execute(self, run: neptune.Run) -> None:
        for epoch_id in tqdm.tqdm(range(self.num_epochs), desc="Training progress"):
            tracker = Tracker()
            with tracker.timer("t_overall"):
                if epoch_id % self.evaluate_every_n_epochs == 0:
                    with tracker.scope("eval"):
                        with tracker.timer("t_overall"):
                            with torch.no_grad():
                                self.evaluate(tracker, epoch_id=epoch_id)

                with tracker.scope("collect_samples"):
                    with tracker.timer("t_overall"):
                        with torch.no_grad():
                            samples = self.collect_samples(tracker=tracker, epoch_id=epoch_id)

                with tracker.scope("train"):
                    with tracker.timer("t_overall"):
                        self.train(samples=samples, tracker=tracker)

            metrics = tracker.report()
            print(
                f"Epoch: {epoch_id}, Loss: {metrics['train/loss_mean']}, "
                f"Total time: {metrics['t_overall_mean']} seconds"
            )
            for key, value in metrics.items():
                run[key].append(value, step=epoch_id)
