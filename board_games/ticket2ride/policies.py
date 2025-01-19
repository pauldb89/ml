import abc
import random

from board_games.ticket2ride.board_logic import Board, Player, RouteInfo
from board_games.ticket2ride.data_models import Card, Tickets, DrawnTickets
from board_games.ticket2ride.policy_helpers import get_valid_actions, get_build_route_options, \
    read_option, get_ticket_draw_options, get_draw_card_options, print_state, ActionType


class Policy:
    @abc.abstractmethod
    def choose_action(self, board: Board, player: Player) -> ActionType:
        pass

    @abc.abstractmethod
    def choose_tickets(
        self,
        board: Board,
        player: Player,
        drawn_tickets: DrawnTickets,
        is_initial_turn: bool,
    ) -> Tickets:
        pass

    # Returning None means drawing from the deck.
    @abc.abstractmethod
    def draw_card(self, board: Board, player: Player, can_draw_any: bool) -> Card | None:
        pass

    @abc.abstractmethod
    def build_route(self, board: Board, player: Player) -> RouteInfo:
        pass


class UniformRandomPolicy(Policy):
    def choose_action(self, board: Board, player: Player) -> ActionType:
        return random.choice(get_valid_actions(board=board, player=player))

    def choose_tickets(
        self,
        board: Board,
        player: Player,
        drawn_tickets: DrawnTickets,
        is_initial_turn: bool,
    ) -> Tickets:
        del board, player
        return random.choice(
            get_ticket_draw_options(tickets=drawn_tickets, is_initial_turn=is_initial_turn)
        )

    def draw_card(self, board: Board, player: Player, can_draw_any: bool) -> Card | None:
        del player
        card_options = get_draw_card_options(board, can_draw_any=can_draw_any)
        return random.choice(card_options)

    def build_route(self, board: Board, player: Player) -> RouteInfo:
        route_options = get_build_route_options(board, player)
        return random.choice(route_options)


class KeyboardInputPolicy(Policy):
    def choose_action(self, board: Board, player: Player) -> ActionType:
        print_state(board=board, player=player)
        valid_actions = get_valid_actions(board=board, player=player)
        return read_option(description="Available actions:", options=valid_actions)

    def choose_tickets(
        self,
        board: Board,
        player: Player,
        drawn_tickets: DrawnTickets,
        is_initial_turn: bool,
    ) -> Tickets:
        print_state(board=board, player=player)
        ticket_options = get_ticket_draw_options(
            tickets=drawn_tickets,
            is_initial_turn=is_initial_turn,
        )
        return read_option(description="Available ticket combos:", options=ticket_options)

    def draw_card(self, board: Board, player: Player, can_draw_any: bool) -> Card | None:
        print_state(board=board, player=player)
        draw_options = get_draw_card_options(board, can_draw_any=can_draw_any)
        return read_option(description="Available draw options:", options=draw_options)

    def build_route(self, board: Board, player: Player) -> RouteInfo:
        print_state(board=board, player=player)
        build_options = get_build_route_options(board, player)
        return read_option(description="Available route building options:", options=build_options)
