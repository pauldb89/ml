import abc
import random

from board_games.ticket2ride.board_logic import Board, Player, RouteInfo
from board_games.ticket2ride.entities import Card, Tickets, DrawnTickets
from board_games.ticket2ride.model import Model
from board_games.ticket2ride.policy_helpers import get_valid_actions, get_build_route_options, \
    read_option, get_ticket_draw_options, get_draw_card_options, print_state, ActionType, \
    ObservedState, DrawCard, Action, Plan, DrawTickets, BuildRoute


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
            next_action_type=random.choice(valid_action_types)
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        card_options = get_draw_card_options(state)
        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=random.choice(card_options)
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
        )

    def build_route(self, state: ObservedState) -> BuildRoute:
        route_options = get_build_route_options(state)
        return BuildRoute(
            player_id=state.player.id,
            action_type=ActionType.BUILD_ROUTE,
            route_info=random.choice(route_options)
        )


class KeyboardInputPolicy(Policy):
    def plan(self, state: ObservedState) -> Plan:
        print_state(state)
        valid_actions = get_valid_actions(state)
        return Plan(
            player_id=state.player.id,
            action_type=ActionType.PLAN,
            next_action_type=read_option(description="Available actions:", options=valid_actions)
        )

    def draw_card(self, state: ObservedState) -> DrawCard:
        print_state(state)
        draw_options = get_draw_card_options(state)
        return DrawCard(
            player_id=state.player.id,
            action_type=ActionType.DRAW_CARD,
            card=read_option(description="Available draw options:", options=draw_options),
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
        )


class ModelPolicy(Policy):
    def __init__(self, model: Model):
        self.model = model

    def plan(self, state: ObservedState) -> Plan:
        return self.model.plan(state, valid_action_types=get_valid_actions(state))

    def draw_card(self, state: ObservedState) -> DrawCard:
        return self.model.draw_card(state, draw_options=get_draw_card_options(state))

    def choose_tickets(self, state: ObservedState) -> DrawTickets:
        return self.model.choose_tickets(state)

    def build_route(self, state: ObservedState) -> BuildRoute:
        return self.model.build_route(state, build_options=get_build_route_options(state))
