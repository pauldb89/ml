from typing import NamedTuple

import torch


class ReplayBatch(NamedTuple):
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminal_states: torch.Tensor


class BasicReplayBuffer:
    def __init__(self, buffer_max_size: int):
        self.buffer_max_size = buffer_max_size

        self.states = torch.empty(0)
        self.next_states = torch.empty(0)
        self.actions = torch.empty(0)
        self.rewards = torch.empty(0)
        self.terminal_states = torch.empty(0)

    def _append(self, buffer: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return torch.cat([buffer, data], dim=0)[-self.buffer_max_size:]

    def add(self, batch: ReplayBatch) -> None:
        self.states = self._append(self.states, batch.states)
        self.next_states = self._append(self.next_states, batch.next_states)
        self.actions = self._append(self.actions, batch.actions)
        self.rewards = self._append(self.rewards, batch.rewards)
        self.terminal_states = self._append(self.terminal_states, batch.terminal_states)

    def sample(self, batch_size: int) -> ReplayBatch:
        buffer_size = self.states.size(0)
        assert batch_size <= buffer_size, f"Batch size {batch_size} > buffer size {buffer_size} "

        indices = torch.randperm(buffer_size)[:batch_size]
        return ReplayBatch(
            states=self.states[indices],
            next_states=self.next_states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            terminal_states=self.terminal_states[indices],
        )
