import abc
import collections
import random
from typing import Protocol

import torch

from board_games.ticket2ride.action_utils import create_build_route_action
from board_games.ticket2ride.action_utils import create_build_route_mask
from board_games.ticket2ride.action_utils import create_draw_card_action
from board_games.ticket2ride.action_utils import create_draw_card_mask
from board_games.ticket2ride.action_utils import create_draw_tickets_action
from board_games.ticket2ride.action_utils import create_draw_tickets_mask
from board_games.ticket2ride.action_utils import create_plan_action
from board_games.ticket2ride.action_utils import create_valid_actions_mask
from board_games.ticket2ride.actions import (
    Action,
    ActionType,
    BuildRoute,
    DrawCard,
    DrawTickets,
    Plan,
)
from board_games.ticket2ride.action_utils import (
    get_build_route_options,
    get_draw_card_options,
    get_ticket_draw_options,
    get_valid_actions,
)
from board_games.ticket2ride.actions import Prediction
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.render_utils import print_state, read_option
from board_games.ticket2ride.state import ObservedState


class Policy(Protocol):
    def choose_action(self, state: ObservedState) -> Action:
        ...

    def choose_actions(self, state: list[ObservedState]) -> list[Action]:
        ...


class ElementPolicy(Policy):
    def choose_action(self, state: ObservedState) -> Action:
        match state.next_action:
            case ActionType.PLAN:
                return self.plan(state)
            case ActionType.DRAW_CARD:
                return self.draw_card(state)
            case ActionType.DRAW_TICKETS:
                return self.choose_tickets(state)
            case ActionType.BUILD_ROUTE:
                return self.build_route(state)
            case _:
                raise ValueError(f"Unsupported action {state.next_action}")

    def choose_actions(self, states: list[ObservedState]) -> list[Action]:
        actions = []
        for state in states:
            actions.append(self.choose_action(state))
        return actions

    @abc.abstractmethod
    def plan(self, state: ObservedState) -> Plan:
        pass

    # Returning None means drawing from the deck.
    @abc.abstractmethod
    def draw_card(self, state: ObservedState) -> DrawCard:
        pass

    @abc.abstractmethod
    def choose_tickets(self, state: ObservedState) -> DrawTickets:
        pass

    @abc.abstractmethod
    def build_route(self, state: ObservedState) -> BuildRoute:
        pass


class UniformRandomPolicy(ElementPolicy):
    def __init__(self, seed: int) -> None:
        super().__init__()

        self.rng = random.Random(seed)

    def plan(self, state: ObservedState) -> Plan:
        valid_action_types = get_valid_actions(state)
        return Plan(
            player_id=state.player.id,
            action_type=ActionType.PLAN,
            next_action_type=self.rng.choice(valid_action_types),
            prediction=None,
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        card_options = get_draw_card_options(state.board, state.consecutive_card_draws)
        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=self.rng.choice(card_options),
            prediction=None,
        )

    def choose_tickets(self, state: ObservedState) -> DrawTickets:
        draw_options = get_ticket_draw_options(
            tickets=state.drawn_tickets,
            is_initial_turn=state.turn_id == 0
        )
        return DrawTickets(
            player_id=state.player.id,
            action_type=ActionType.DRAW_TICKETS,
            tickets=self.rng.choice(draw_options),
            prediction=None,
        )

    def build_route(self, state: ObservedState) -> BuildRoute:
        route_options = get_build_route_options(state)
        return BuildRoute(
            player_id=state.player.id,
            action_type=ActionType.BUILD_ROUTE,
            route_info=self.rng.choice(route_options),
            prediction=None,
        )


class KeyboardInputPolicy(ElementPolicy):
    def plan(self, state: ObservedState) -> Plan:
        print_state(state)
        valid_actions = get_valid_actions(state)
        return Plan(
            player_id=state.player.id,
            action_type=ActionType.PLAN,
            next_action_type=read_option(description="Available actions:", options=valid_actions),
            prediction=None,
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        print_state(state)
        draw_options = get_draw_card_options(state.board, state.consecutive_card_draws)
        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=read_option(description="Available draw options:", options=draw_options),
            prediction=None,
        )

    def choose_tickets(self, state: ObservedState) -> DrawTickets:
        print_state(state)
        ticket_options = get_ticket_draw_options(
            tickets=state.drawn_tickets,
            is_initial_turn=state.turn_id == 0,
        )
        return DrawTickets(
            player_id=state.player.id,
            action_type=ActionType.DRAW_TICKETS,
            tickets=read_option(description="Available ticket combos:", options=ticket_options),
            prediction=None,
        )

    def build_route(self, state: ObservedState) -> BuildRoute:
        print_state(state)
        build_options = get_build_route_options(state)
        return BuildRoute(
            player_id=state.player.id,
            action_type=ActionType.BUILD_ROUTE,
            route_info=read_option(
                description="Available route building options:",
                options=build_options,
            ),
            prediction=None,
        )


class ModelPolicy(Policy):
    def __init__(self, model: Model):
        self.model = model

    def choose_action(self, state: ObservedState) -> Action:
        return self.choose_actions([state])[0]

    def choose_actions(self, states: list[ObservedState]) -> list[Action]:
        grouped_states = collections.defaultdict(list)
        action_index = []
        for state in states:
            action_index.append((state.next_action, len(grouped_states[state.next_action])))
            grouped_states[state.next_action].append(state)

        grouped_actions = {
            action_type: self.infer_actions(action_type, states)
            for action_type, states in grouped_states.items()
        }

        return [grouped_actions[action_type][idx] for action_type, idx in action_index]

    def infer_actions(self, action_type: ActionType, states: list[ObservedState]) -> list[Action]:
        logits, values = self.model(states=states, head=action_type, mask=self.create_mask(action_type, states))
        class_ids = self.classify(logits)
        log_probs = torch.log_softmax(logits, dim=-1)
        class_log_probs = log_probs.gather(dim=1, index=class_ids.unsqueeze(dim=1)).squeeze(dim=1)
        return self.create_actions(
            action_type,
            states,
            class_ids=class_ids.detach().cpu().numpy().tolist(),
            log_probs=class_log_probs.detach().cpu().numpy().tolist(),
            values=values.detach().cpu().numpy().tolist(),
        )

    def create_mask(self, action_type: ActionType, states: list[ObservedState]) -> torch.Tensor:
        mask_fns = {
            ActionType.PLAN: create_valid_actions_mask,
            ActionType.DRAW_CARD: create_draw_card_mask,
            ActionType.DRAW_TICKETS: create_draw_tickets_mask,
            ActionType.BUILD_ROUTE: create_build_route_mask,
        }
        mask_fn = mask_fns[action_type]
        return torch.tensor([mask_fn(state) for state in states])

    def create_actions(
        self,
        action_type: ActionType,
        states: list[ObservedState],
        class_ids: list[int],
        log_probs: list[float],
        values: list[float],
    ) -> list[Action]:
        create_fns = {
            ActionType.PLAN: create_plan_action,
            ActionType.DRAW_CARD: create_draw_card_action,
            ActionType.DRAW_TICKETS: create_draw_tickets_action,
            ActionType.BUILD_ROUTE: create_build_route_action,
        }
        create_fn = create_fns[action_type]
        return [
            create_fn(state, Prediction(class_id=class_id, log_prob=log_prob, value=value))
            for state, class_id, log_prob, value in zip(states, class_ids, log_probs, values)
        ]

    @abc.abstractmethod
    def classify(self, logits: torch.Tensor) -> torch.Tensor:
        ...


class StochasticModelPolicy(ModelPolicy):
    def classify(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=logits).sample()


class ArgmaxModelPolicy(ModelPolicy):
    def classify(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)
