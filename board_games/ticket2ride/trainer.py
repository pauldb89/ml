import abc
import collections
import copy
import hashlib
import json
import os
import random

import numpy as np
import torch
import tqdm
import wandb

from torch.cuda.amp import GradScaler

from board_games.ticket2ride.actions import ActionType, Action, DrawCard
from board_games.ticket2ride.color import ANY
from board_games.ticket2ride.environment import Environment, Roller
from board_games.ticket2ride.model import Model, RawSample, Sample
from board_games.ticket2ride.policies import Policy, UniformRandomPolicy, \
    ArgmaxModelPolicy, StochasticModelPolicy
from board_games.ticket2ride.render_utils import print_state
from board_games.ticket2ride.state import PlayerScore
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
                    episode_id=raw_sample.episode_id,
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
                episode_id=last_sample.episode_id,
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
                    episode_id=raw_sample.episode_id,
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
                episode_id=initial_sample.episode_id,
                state=initial_sample.state,
                action=initial_sample.action,
                score=initial_sample.score,
                reward=reward,
            )
            samples.append(sample)

        return samples


def track_action_stats(tracker: Tracker, actions: list[Action]) -> None:
    new_turn = True
    consecutive_draws = 0
    longest_draw = 0
    draw_lengths = collections.defaultdict(int)
    card_draws_deck = 0
    card_draws_color = 0
    card_draws_any = 0
    for action in actions:
        if isinstance(action, DrawCard):
            if action.card is None:
                card_draws_deck += 1
            elif action.card.color == ANY:
                card_draws_any += 1
            else:
                card_draws_color += 1

        if new_turn:
            if action.action_type == ActionType.DRAW_CARD:
                consecutive_draws += 1
            else:
                if consecutive_draws:
                    draw_lengths[consecutive_draws] += 1
                consecutive_draws = 0

        longest_draw = max(longest_draw, consecutive_draws)
        new_turn = action.action_type == ActionType.PLAN

    tracker.log_value("longest_draw", longest_draw)
    tracker.log_value("card_draws_any", card_draws_any)
    tracker.log_value("card_draws_deck", card_draws_deck)
    tracker.log_value("card_draws_color", card_draws_color)
    for length, cnt in draw_lengths.items():
        tracker.log_value(f"consecutive_draws_len_{length:02d}", cnt)


def track_score_stats(tracker: Tracker, score: PlayerScore) -> None:
    tracker.log_value("total_points", score.total_points)
    tracker.log_value("route_points", score.route_points)
    tracker.log_value("tickets_drawn", score.total_tickets)
    tracker.log_value("tickets_completed", score.completed_tickets)
    tracker.log_value("ticket_points", score.ticket_points)
    tracker.log_value("longest_path_bonus", score.longest_path_bonus)
    tracker.log_value("completed_routes", sum(score.owned_routes_by_length.values()))
    for length in range(1, 7):
        num_routes = score.owned_routes_by_length[length]
        tracker.log_value(f"completed_routes_len_{length}", num_routes)
        for _ in range(num_routes):
            tracker.log_value(f"completed_routes_length", length)


def normalize_rewards(samples: list[Sample]) -> list[Sample]:
    rewards = [s.reward for s in samples]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    for s in samples:
        s.reward = (s.reward - mean_reward) / (std_reward + 1e-6)

    return samples


def tensor_hash(t: torch.Tensor) -> str:
    data = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class PolicyGradientTrainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        num_players: int,
        num_epochs: int,
        num_samples_per_epoch: int,
        num_eval_samples_per_epoch: int,
        batch_size: int,
        evaluate_every_n_epochs: int,
        checkpoint_every_n_epochs: int,
        reward_fn: Reward,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.num_players = num_players
        self.num_epochs = num_epochs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.num_eval_samples_per_epoch = num_eval_samples_per_epoch
        self.batch_size = batch_size
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.reward_fn = reward_fn

        self.scaler = GradScaler(init_scale=2.**16)

        self.episode_id = 0

    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()
        policy = StochasticModelPolicy(model=self.model)

        envs = []
        states = []
        episodes = list(range(self.num_samples_per_epoch))
        for episode_id in episodes:
            env = Environment(num_players=self.num_players)
            envs.append(env)
            states.append(env.reset(epoch_id * self.num_samples_per_epoch + episode_id))

        raw_samples: list[dict[int, list[RawSample]]] = [collections.defaultdict(list) for _ in episodes]
        terminal_transitions: list[Transition | None] = [None for _ in episodes]
        while episodes:
            with tracker.timer("t_policy_choose_action"):
                actions = policy.choose_actions([states[episode_id] for episode_id in episodes])

            active_episodes = []
            for episode_id, action in zip(episodes, actions):
                with tracker.timer("t_env_step"):
                    transition = envs[episode_id].step(action)

                raw_samples[episode_id][action.player_id].append(
                    RawSample(episode_id=episode_id, state=states[episode_id], action=action, score=transition.score)
                )

                if transition.state.terminal:
                    terminal_transitions[episode_id] = transition
                else:
                    active_episodes.append(episode_id)

                states[episode_id] = transition.state

            episodes = active_episodes

        for transition in terminal_transitions:
            for score in transition.score.scorecard:
                track_score_stats(tracker, score)

        samples = []
        for per_episode_samples, terminal_transition in zip(raw_samples, terminal_transitions):
            for per_player_samples in per_episode_samples.values():
                actions = [sample.action for sample in per_player_samples]
                track_action_stats(tracker, actions)

            episode_length = 0
            with tracker.timer("t_compute_rewards"):
                for per_player_samples in per_episode_samples.values():
                    assert terminal_transition is not None
                    episode_length += len(per_player_samples)
                    samples.extend(self.reward_fn.apply(per_player_samples, terminal_transition, epoch_id=epoch_id))

            tracker.log_value("episode_length", episode_length)

        action_counts = collections.defaultdict(int)
        for s in samples:
            action_counts[(s.action.action_type, s.action.class_id)] += 1
        tracker.log_value("unique_explored_actions", len(action_counts))

        return normalize_rewards(samples)

    def train(self, samples: list[Sample], tracker: Tracker, epoch_id: int) -> None:
        self.model.train()

        self.optimizer.zero_grad()

        num_batches = 0
        losses = []
        for start_index in tqdm.tqdm(range(0, len(samples), self.batch_size), desc="Train step"):
            num_batches += 1
            batch_samples = samples[start_index:start_index + self.batch_size]

            with torch.cuda.amp.autocast():
                loss = len(batch_samples) / len(samples) * self.model.loss(batch_samples)
                assert torch.isnan(loss).sum() == 0
                losses.append(loss)
                loss = self.scaler.scale(loss)
                loss.backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        total_loss = sum(losses)
        loss_hash = tensor_hash(total_loss)
        wandb.log({"loss_hash": loss_hash}, step=epoch_id)
        tracker.log_value("loss", total_loss.item())
        tracker.log_value("num_batches", num_batches)

    def evaluate(self, tracker: Tracker, epoch_id: int) -> None:
        self.model.eval()

        model_policy = ArgmaxModelPolicy(model=self.model)
        random_policies = [UniformRandomPolicy(seed=0) for _ in range(self.num_players - 1)]

        with tracker.scope("vs_random"):
            for idx in tqdm.tqdm(range(self.num_eval_samples_per_epoch), desc="Eval step"):
                index = random.randint(0, len(random_policies))
                policies: list[Policy] = copy.deepcopy(random_policies)
                policies.insert(index, model_policy)
                roller = Roller(env=Environment(num_players=self.num_players), policies=policies)
                stats = roller.run(seed=idx)

                actions = []
                for t in stats.transitions:
                    if t.action.player_id == index:
                        actions.append(t.action)

                track_action_stats(tracker, actions)

                score = stats.transitions[-1].score
                tracker.log_value("wins", score.winner_id == index)
                track_score_stats(tracker, score.scorecard[index])

        eval_metrics = {}
        for key, value in tracker.report().items():
            if key.startswith("eval/vs_random") and key.endswith("mean"):
                eval_metrics[key] = value

        print(f"Eval metrics at epoch {epoch_id}: {json.dumps(eval_metrics, indent=2, sort_keys=True)}")

    def checkpoint(self, epoch_id: int) -> None:
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f"model_{epoch_id:05d}.path"))

    def execute(self) -> None:
        for epoch_id in tqdm.tqdm(range(self.num_epochs), desc="Training progress"):
            tracker = Tracker()
            with tracker.timer("t_overall"):
                if epoch_id % self.checkpoint_every_n_epochs == 0:
                    with tracker.scope("checkpoint"):
                        with tracker.timer("t_overall"):
                            self.checkpoint(epoch_id=epoch_id)

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
                        self.train(samples=samples, tracker=tracker, epoch_id=epoch_id)

            metrics = tracker.report()
            print(
                f"Epoch: {epoch_id}, Loss: {metrics['train/loss_mean']}, "
                f"Total time: {metrics['t_overall_mean']} seconds"
            )
            wandb.log(metrics, step=epoch_id)

        tracker = Tracker()
        with tracker.scope("checkpoint"):
            with tracker.timer("t_overall"):
                self.checkpoint(epoch_id=self.num_epochs)

        with tracker.scope("eval"):
            with tracker.timer("t_overall"):
                with torch.no_grad():
                    self.evaluate(tracker, epoch_id=self.num_epochs)
        metrics = tracker.report()
        wandb.log(metrics, step=self.num_epochs)
