import itertools
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch

from wordle.discount_schedule import DiscountSchedule
from wordle.environment import WordleEnv
from wordle.model import TransformerPolicy


Episode = List[Dict[str, Any]]


class WordleSolver:
    def __init__(
        self,
        train_env: WordleEnv,
        eval_env: WordleEnv,
        policy: TransformerPolicy,
        optimizer: torch.optim.Optimizer,
        discount_schedule: DiscountSchedule,
        rewards: Dict[str, float],
        reshape_rewards_max_iteration: Optional[int],
        iterations: int,
        episodes_per_iteration: int,
        steps_per_iteration: int,
        batch_size: int,
        evaluate_every_n_iterations: int,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.policy = policy
        self.optimizer = optimizer
        self.discount_schedule = discount_schedule
        self.rewards = rewards
        self.reshape_rewards_max_iteration = reshape_rewards_max_iteration
        self.iterations = iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.steps_per_iteration = steps_per_iteration
        self.batch_size = batch_size
        self.evaluate_every_n_iterations = evaluate_every_n_iterations

    def collect_data(self) -> List[Episode]:
        self.policy.eval()

        episodes = []
        for episode_idx in range(self.episodes_per_iteration):
            state, _ = self.train_env.reset()
            terminal_state = False
            episode = []

            while not terminal_state:
                action = self.policy.predict(state)
                next_state, reward, terminal_state, _, _ = self.train_env.step(action)
                episode.append({"state": state, "reward": reward, "action": action})

            episodes.append(episode)

        self.policy.train()

        return episodes

    def reshape_rewards(self, episodes: List[Episode], iteration: int) -> List[Episode]:
        if self.reshape_rewards_max_iteration is not None and iteration > self.reshape_rewards_max_iteration:
            return episodes

        for episode in episodes:
            cumulated_reward = 0
            for turn in episode:
                if turn["reward"] == 0:
                    _, last_match_state = turn["states"][-1]
                    new_reward = sum(self.rewards[letter_state] for letter_state in last_match_state)
                    cumulated_reward += new_reward
                    turn["reward"] = new_reward
                else:
                    turn["reward"] -= cumulated_reward

        return episodes

    def update_rewards(self, episodes: List[Episode], iteration: int) -> List[Episode]:
        episodes = self.reshape_rewards(episodes=episodes, iteration=iteration)

        discount_factor = self.discount_schedule.get(iteration)
        for episode in episodes:
            cumulated_reward = 0
            for turn in reversed(episode):
                cumulated_reward = cumulated_reward * discount_factor + turn["reward"]
                turn["reward"] = cumulated_reward

        return episodes

    def train_step(self, episodes: List[Episode]) -> None:
        examples = list(itertools.chain(*episodes))
        for step in range(self.steps_per_iteration):
            batch = random.sample(examples, self.batch_size)

            input_states = [example["states"] for example in batch]
            actions = [example["action"] for example in batch]
            rewards = [example["reward"] for example in batch]

            loss = self.policy(input_states=input_states, actions=actions, rewards=rewards)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def execute(self) -> None:
        iteration = 0
        while iteration < self.iterations:
            iteration += 1

            episodes = self.collect_data()

            episodes = self.update_rewards(episodes=episodes, iteration=iteration)

            self.train_step(episodes=episodes)

            if iteration % self.evaluate_every_n_iterations:
                # TODO(pauldb): Implement evaluation
                pass
