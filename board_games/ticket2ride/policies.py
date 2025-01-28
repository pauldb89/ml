import abc
import random

import torch

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
    get_valid_actions, PLAN_CLASSES, DRAW_CARD_CLASSES, CHOOSE_TICKETS_CLASSES, BUILD_ROUTE_CLASSES,
)
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.render_utils import print_state, read_option
from board_games.ticket2ride.route import ROUTES
from board_games.ticket2ride.route_info import RouteInfo
from board_games.ticket2ride.state import ObservedState


class Policy:
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


class UniformRandomPolicy(Policy):
    def plan(self, state: ObservedState) -> Plan:
        valid_action_types = get_valid_actions(state)
        return Plan(
            player_id=state.player.id,
            action_type=ActionType.PLAN,
            next_action_type=random.choice(valid_action_types),
            class_id=None,
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        card_options = get_draw_card_options(state)
        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=random.choice(card_options),
            class_id=None,
        )

    def choose_tickets(self, state: ObservedState) -> DrawTickets:
        draw_options = get_ticket_draw_options(
            tickets=state.drawn_tickets,
            is_initial_turn=state.turn_id == 0
        )
        return DrawTickets(
            player_id=state.player.id,
            action_type=ActionType.DRAW_TICKETS,
            tickets=random.choice(draw_options),
            class_id=None,
        )

    def build_route(self, state: ObservedState) -> BuildRoute:
        route_options = get_build_route_options(state)
        return BuildRoute(
            player_id=state.player.id,
            action_type=ActionType.BUILD_ROUTE,
            route_info=random.choice(route_options),
            class_id=None,
        )


class KeyboardInputPolicy(Policy):
    def plan(self, state: ObservedState) -> Plan:
        print_state(state)
        valid_actions = get_valid_actions(state)
        return Plan(
            player_id=state.player.id,
            action_type=ActionType.PLAN,
            next_action_type=read_option(description="Available actions:", options=valid_actions),
            class_id=None,
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        print_state(state)
        draw_options = get_draw_card_options(state)
        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=read_option(description="Available draw options:", options=draw_options),
            class_id=None,
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
            class_id=None,
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
            class_id=None,
        )


class ModelPolicy(Policy):
    def __init__(self, model: Model):
        self.model = model

    @abc.abstractmethod
    def classify(self, logits: torch.Tensor) -> int:
        return torch.argmax(logits).item()

    def plan(self, state: ObservedState) -> Plan:
        valid_action_types = get_valid_actions(state)
        mask = []
        for action_type in PLAN_CLASSES:
            mask.append(int(action_type in valid_action_types))

        logits = self.model(
            states=[state],
            head=ActionType.PLAN,
            mask=torch.tensor([mask]),
        ).squeeze(0)

        class_id = self.classify(logits)

        return Plan(
            player_id=state.player.id,
            action_type=ActionType.PLAN,
            next_action_type=PLAN_CLASSES[class_id],
            class_id=class_id,
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        draw_options = get_draw_card_options(state)
        mask = []
        for cls in DRAW_CARD_CLASSES:
            mask.append(int(cls in draw_options))

        logits = self.model(
            states=[state],
            head=ActionType.DRAW_CARD,
            mask=torch.tensor([mask]),
        ).squeeze(0)

        class_id = self.classify(logits)

        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=DRAW_CARD_CLASSES[class_id],
            class_id=class_id,
        )

    def choose_tickets(self, state: ObservedState) -> DrawTickets:
        mask = []
        for combo in CHOOSE_TICKETS_CLASSES:
            mask.append(1 if len(combo) >= 2 or state.turn_id > 0 else 0)

        logits = self.model(
            states=[state],
            head=ActionType.DRAW_TICKETS,
            mask=torch.tensor([mask]),
        ).squeeze(0)

        class_id = self.classify(logits)
        combo = CHOOSE_TICKETS_CLASSES[class_id]

        return DrawTickets(
            player_id=state.player.id,
            action_type=ActionType.DRAW_TICKETS,
            tickets=tuple(state.drawn_tickets[ticket_idx] for ticket_idx in combo),
            class_id=class_id,
        )

    def build_route(self, state: ObservedState) -> BuildRoute:
        build_options = get_build_route_options(state)

        valid_options = set()
        for route_info in build_options:
            valid_options.add((ROUTES[route_info.route_id], route_info.color))

        mask = []
        for cls in BUILD_ROUTE_CLASSES:
            mask.append(int(cls in valid_options))

        logits = self.model(
            states=[state],
            head=ActionType.BUILD_ROUTE,
            mask=torch.tensor([mask]),
        ).squeeze(0)

        class_id = self.classify(logits)
        route, color = BUILD_ROUTE_CLASSES[class_id]

        return BuildRoute(
            player_id=state.player.id,
            action_type=ActionType.BUILD_ROUTE,
            route_info=RouteInfo(
                route_id=route.id,
                player_id=state.player.id,
                color=color,
                num_any_cards=max(0, route.length - state.player.card_counts[color]),
            ),
            class_id=class_id,
        )


class StochasticModelPolicy(ModelPolicy):
    def classify(self, logits: torch.Tensor) -> int:
        return torch.distributions.Categorical(logits=logits).sample().item()


class ArgmaxModelPolicy(ModelPolicy):
    def classify(self, logits: torch.Tensor) -> int:
        return torch.argmax(logits, dim=-1).item()
