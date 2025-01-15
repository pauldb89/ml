import abc
import itertools
import random

from board_games.ticket2ride.board_logic import Board, Player
from board_games.ticket2ride.consts import ANY
from board_games.ticket2ride.data_models import Ticket, Card, RouteInfo, ActionType
from board_games.ticket2ride.policy_helpers import get_valid_actions, get_build_route_options


class Policy:
    @abc.abstractmethod
    def choose_action(self, board: Board, player: Player) -> ActionType:
        pass

    @abc.abstractmethod
    def choose_tickets(
        self,
        board: Board,
        player: Player,
        ticket_options: list[Ticket],
        is_initial_turn: bool,
    ) -> list[Ticket]:
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
        ticket_options: list[Ticket],
        is_initial_turn: bool,
    ) -> list[Ticket]:
        draw_options = [
            ticket_options,
            *itertools.combinations(ticket_options, 2),
        ]
        if not is_initial_turn:
            draw_options.extend(itertools.combinations(ticket_options, 1))

        return random.choice(draw_options)

    def draw_card(self, board: Board, player: Player, can_draw_any: bool) -> Card | None:
        card_options: list[Card | None] = []
        if len(board.card_deck) >= 1:
            card_options.append(None)

        for card in board.visible_cards:
            if card in card_options:
                continue

            if card.color == ANY and not can_draw_any:
                continue

            card_options.append(card)

        return random.choice(card_options)

    def build_route(self, board: Board, player: Player) -> RouteInfo:
        route_options = get_build_route_options(board, player)
        return random.choice(route_options)
