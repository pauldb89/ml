import abc
from argparse import Action
import torch
from board_games.ticket2ride.environment import Environment
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.state import ObservedState, Transition, Score


class PolicyGradientSolver:
    def __init__(
        self,
        env: Environment,
        model: Model,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_samples_per_epoch: int,
        discount: float,
    ):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.discount = discount

    def collect_data(self) -> None:
        self.model.eval()

        states: list[ObservedState] = []
        actions: list[Action] = []
        scores: list[Score] = []
        terminal_states: list[bool] = []

        # TODO(pauldb): Consider the implications of determinism.
        for _ in range(self.num_samples_per_epoch):
            terminal_state = False
            state, _ = self.env.reset()

            while not terminal_state:
                action = self.model.choose_action(state)
                transition = self.env.step(action)

                states.append(state)
                actions.append(action)
                scores.append(transition.score)
                terminal_states.append(transition.state.terminal)

                if transition.state.terminal:
                    state = self.env.reset()
                else:
                    state = transition.state




    def execute(self) -> None:
        for epoch in range(self.num_epochs):
            train_data = self.collect_data()
