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
from board_games.ticket2ride.environment import BatchRoller
from board_games.ticket2ride.model import Model, Sample
from board_games.ticket2ride.policies import UniformRandomPolicy, ArgmaxModelPolicy, StochasticModelPolicy
from board_games.ticket2ride.state import PlayerScore
from board_games.ticket2ride.state import Transition
from board_games.ticket2ride.tracker import Tracker


class Reward:
    def __init__(
        self,
        win_reward: float,
        initial_draw_card_reward: float,
        final_draw_card_reward: float,
        draw_card_horizon_epochs: int,
        reward_scale: float,
    ) -> None:
        super().__init__()
        self.win_reward = win_reward
        self.initial_draw_card_reward = initial_draw_card_reward
        self.final_draw_card_reward = final_draw_card_reward
        self.draw_card_horizon_epochs = draw_card_horizon_epochs
        self.reward_scale = reward_scale

    def apply(
        self,
        player_samples: list[Sample],
        final_transition: Transition,
        epoch_id: int,
    ) -> list[Sample]:
        samples = []
        rate = min(epoch_id / self.draw_card_horizon_epochs, 1)
        draw_card_bonus = (1 - rate) * self.initial_draw_card_reward + rate * self.final_draw_card_reward
        for idx, raw_sample in enumerate(reversed(player_samples)):
            reward = raw_sample.score.turn_score.total_points
            if raw_sample.action.action_type == ActionType.DRAW_CARD:
                reward += draw_card_bonus
            if idx == 0:
                if final_transition.score.winner_id == raw_sample.action.player_id:
                    reward += self.win_reward
                else:
                    reward -= self.win_reward

            samples.append(
                Sample(
                    episode_id=raw_sample.episode_id,
                    state=raw_sample.state,
                    action=raw_sample.action,
                    score=raw_sample.score,
                    reward=reward * self.reward_scale,
                )
            )

        return list(reversed(samples))


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
        num_episodes_per_epoch: int,
        num_eval_episodes_per_epoch: int,
        batch_size: int,
        evaluate_every_n_epochs: int,
        checkpoint_every_n_epochs: int,
        value_loss_weight: float,
        reward_fn: Reward,
        reward_discount: float,
        gae_lambda: float,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.num_players = num_players
        self.num_epochs = num_epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.batch_size = batch_size
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.value_loss_weight = value_loss_weight
        self.reward_fn = reward_fn
        # Often denoted as gamma, used to compute long term reward from a particular state:
        # G_t = r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + ...
        self.reward_discount = reward_discount
        self.gae_lambda = gae_lambda

        self.scaler = GradScaler(init_scale=2.**16)

        self.episode_id = 0

    def compute_advantages(self, player_samples: list[Sample]) -> list[Sample]:
        # next_state_value = terminal_value
        next_state_value = 0
        long_term_return = 0
        advantage = 0
        advantages = []
        for sample in reversed(player_samples):
            sample.long_term_return = sample.reward + long_term_return * self.reward_discount

            current_value = sample.action.prediction.value
            td_error = sample.reward + self.reward_discount * next_state_value - current_value
            sample.advantage = td_error + advantage * self.reward_discount * self.gae_lambda
            sample.estimated_return = sample.advantage + current_value

            next_state_value = current_value
            long_term_return = sample.long_term_return
            advantage = sample.advantage
            advantages.append(advantage)

        mean, std = np.mean(advantages), np.std(advantages)
        for sample in player_samples:
            sample.advantage = (sample.advantage - mean) / (std + 1e-8)

        return player_samples

    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()
        # envs = []
        # states = []
        # episodes = list(range(self.num_samples_per_epoch))
        # for episode_id in episodes:
        #     env = Environment(num_players=self.num_players)
        #     envs.append(env)
        #     states.append(env.reset(epoch_id * self.num_samples_per_epoch + episode_id))
        #
        # raw_samples: list[dict[int, list[Sample]]] = [collections.defaultdict(list) for _ in episodes]
        # terminal_transitions: list[Transition | None] = [None for _ in episodes]
        # while episodes:
        #     with tracker.timer("t_policy_choose_action"):
        #         actions = policy.choose_actions([states[episode_id] for episode_id in episodes])
        #
        #     active_episodes = []
        #     for episode_id, action in zip(episodes, actions):
        #         with tracker.timer("t_env_step"):
        #             transition = envs[episode_id].step(action)
        #
        #         raw_samples[episode_id][action.player_id].append(
        #             Sample(episode_id=episode_id, state=states[episode_id], action=action, score=transition.score)
        #         )
        #
        #         if transition.state.terminal:
        #             terminal_transitions[episode_id] = transition
        #         else:
        #             active_episodes.append(episode_id)
        #
        #         states[episode_id] = transition.state
        #
        #     episodes = active_episodes
        roller = BatchRoller()
        episode_ids = [epoch_id * self.num_episodes_per_epoch + idx for idx in range(self.num_episodes_per_epoch)]
        transitions = roller.run(
            seeds=episode_ids,
            policies=[StochasticModelPolicy(model=self.model)],
            player_policy_ids=[[0] * self.num_players] * self.num_episodes_per_epoch,
            tracker=tracker,
        )

        for episode_transitions in transitions:
            for score in episode_transitions[-1].score.scorecard:
                track_score_stats(tracker, score)

        samples = []
        for episode_id, episode_transitions in zip(episode_ids, transitions):
            per_player_samples = collections.defaultdict(list)
            for transition in episode_transitions:
                per_player_samples[transition.action.player_id].append(
                    Sample(
                        episode_id=episode_id,
                        state=transition.source_state,
                        action=transition.action,
                        score=transition.score,
                    )
                )

            for player_samples in per_player_samples.values():
                actions = [sample.action for sample in player_samples]
                track_action_stats(tracker, actions)

            episode_length = 0
            with tracker.timer("t_compute_rewards"):
                for player_samples in per_player_samples.values():
                    episode_length += len(player_samples)

                    player_samples = self.reward_fn.apply(
                        player_samples,
                        episode_transitions[-1],
                        epoch_id=epoch_id,
                    )
                    player_samples = self.compute_advantages(player_samples)
                    samples.extend(player_samples)

            tracker.log_value("episode_length", episode_length)

        action_counts = collections.defaultdict(int)
        for s in samples:
            action_counts[(s.action.action_type, s.action.prediction.class_id)] += 1
        tracker.log_value("unique_explored_actions", len(action_counts))

        values = np.array([s.action.prediction.value for s in samples])
        returns = np.array([s.long_term_return for s in samples])

        v = values.tolist()
        r = returns.tolist()
        print(f"{v[:10]=}")
        print(f"{r[:10]=}")
        print(f"{v[-10:]=}")
        print(f"{r[-10:]=}")
        tracker.log_value("value_mean_absolute_error", np.mean(np.abs(values - returns)))
        tracker.log_value("value_mean_signed_error", np.mean(values - returns))
        tracker.log_value("value_mean_squared_error", np.mean(np.square(values - returns)))
        tracker.log_value("value_explained_variance", (1 - np.var(values - returns) / (np.var(returns) + 1e-8)))
        tracker.log_value("num_samples", len(samples))

        return samples

    def train(self, samples: list[Sample], tracker: Tracker, epoch_id: int) -> None:
        self.model.train()

        self.optimizer.zero_grad()

        num_batches = 0
        losses = []
        random.shuffle(samples)
        for start_index in tqdm.tqdm(range(0, len(samples), self.batch_size), desc="Train step"):
            num_batches += 1
            batch_samples = samples[start_index:start_index + self.batch_size]

            with torch.cuda.amp.autocast():
                batch_weight = len(batch_samples) / len(samples)
                policy_loss, value_loss = self.model.loss(batch_samples)

                policy_loss *= batch_weight
                tracker.log_value("policy_loss", policy_loss.item())

                value_loss *= batch_weight
                tracker.log_value("value_loss", value_loss.item())

                # loss = policy_loss + self.value_loss_weight * value_loss
                loss = self.value_loss_weight * value_loss
                assert torch.isnan(loss).sum() == 0
                tracker.log_value("loss", loss.item())
                losses.append(loss)

                loss = self.scaler.scale(loss)

            loss.backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        total_loss = sum(losses)
        loss_hash = tensor_hash(total_loss)
        wandb.log({"loss_hash": loss_hash}, step=epoch_id)

        tracker.log_value("num_batches", num_batches)

    def evaluate(self, tracker: Tracker, epoch_id: int) -> None:
        self.model.eval()

        with tracker.scope("vs_random"):
            policies = [ArgmaxModelPolicy(model=self.model), UniformRandomPolicy(seed=0)]
            rng = random.Random(0)
            player_policy_ids = []
            model_policy_indices = []
            for _ in range(self.num_eval_episodes_per_epoch):
                model_policy_index = rng.randint(0, self.num_players - 1)
                model_policy_indices.append(model_policy_index)

                policy_ids = [1] * (self.num_players - 1)
                policy_ids.insert(model_policy_index, 0)
                player_policy_ids.append(policy_ids)

            roller = BatchRoller()
            transitions = roller.run(
                seeds=list(range(self.num_eval_episodes_per_epoch)),
                policies=policies,
                player_policy_ids=player_policy_ids,
                tracker=tracker,
            )

            for episode_transitions, model_policy_index in zip(transitions, model_policy_indices):
                actions = []
                for t in episode_transitions:
                    if t.action.player_id == model_policy_index:
                        actions.append(t.action)

                track_action_stats(tracker, actions)

                score = episode_transitions[-1].score
                tracker.log_value("wins", score.winner_id == model_policy_index)
                track_score_stats(tracker, score.scorecard[model_policy_index])

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
