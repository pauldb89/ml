import itertools
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from tabulate import tabulate
from termcolor import colored

from board_games.ticket2ride.board_logic import Board, Player, check_tickets, RouteInfo
from board_games.ticket2ride.consts import ROUTES, ANY, COLORS
from board_games.ticket2ride.entities import Card, Ticket, DrawnTickets, Tickets, render_cards
from board_games.ticket2ride.longest_path import find_longest_paths


class ActionType(StrEnum):
    DRAW_CARDS = "DRAW_CARDS"
    BUILD_ROUTE = "BUILD_ROUTE"
    DRAW_TICKETS = "DRAW_TICKETS"


@dataclass(frozen=True)
class Action:
    player_id: int
    action_type: ActionType


@dataclass(frozen=True)
class DrawCards(Action):
    cards: list[Card | None]

    def __repr__(self) -> str:
        text_parts = []
        for card in self.cards:
            if card is None:
                text_parts.append("card from deck")
            else:
                text_parts.append(repr(card))

        text = " and ".join(text_parts)
        return f"Player {self.player_id} drew cards: {text}"


@dataclass(frozen=True)
class DrawTickets(Action):
    tickets: Tickets

    def __repr__(self) -> str:
        return f"Player {self.player_id} drew {len(self.tickets)} tickets"


@dataclass(frozen=True)
class BuildRoute(Action):
    route_info: RouteInfo

    def __repr__(self) -> str:
        return f"Player {self.player_id} built {self.route_info}"


def get_build_route_options(board: Board, player: Player) -> list[RouteInfo]:
    route_options: list[RouteInfo] = []
    for route in ROUTES:
        # This route has already been built, so it's not a valid option.
        if route.id in board.route_ownership:
            continue

        # The route is too long for the number of train cars the player currently has left.
        if route.length > board.train_cars[player.id]:
            continue

        # Check if we can use locomotive cards alone to build the route.
        if ANY in player.card_counts and player.card_counts[ANY] >= route.length:
            route_options.append(
                RouteInfo(
                    route_id=route.id,
                    player_id=player.id,
                    color=ANY,
                    num_any_cards=route.length,
                )
            )

        color_options = COLORS if route.color == ANY else [route.color]
        for color in color_options:
            if (
                    color in player.card_counts
                    and player.card_counts[color] + player.card_counts.get(ANY, 0) >= route.length
            ):
                route_options.append(
                    RouteInfo(
                        route_id=route.id,
                        player_id=player.id,
                        color=color,
                        # Greedily first use cards of the given color, then use locomotives (ANY).
                        num_any_cards=max(0, route.length - player.card_counts[color]),
                    )
                )

    return route_options


# TODO(pauldb): Unit test.
def get_valid_actions(board: Board, player: Player) -> list[ActionType]:
    valid_action_types = []
    if len(board.card_deck) + len(board.visible_cards) >= 2:
        valid_action_types.append(ActionType.DRAW_CARDS)

    if len(board.ticket_deck) >= 3:
        valid_action_types.append(ActionType.DRAW_TICKETS)

    build_route_options = get_build_route_options(board, player)
    if len(build_route_options) > 0:
        valid_action_types.append(ActionType.BUILD_ROUTE)

    return valid_action_types


# TODO(pauldb): Unit test.
def get_ticket_draw_options(tickets: DrawnTickets, is_initial_turn: bool) -> list[Tickets]:
    draw_options = [tickets, *itertools.combinations(tickets, 2)]
    if not is_initial_turn:
        draw_options.extend(itertools.combinations(tickets, 1))
    return draw_options


# TODO(pauldb): Unit test.
def get_draw_card_options(board: Board, can_draw_any: bool) -> list[Card | None]:
    card_options: list[Card | None] = []
    if len(board.card_deck) >= 1:
        card_options.append(None)

    for card in board.visible_cards:
        if card in card_options:
            continue

        if card.color == ANY and not can_draw_any:
            continue

        card_options.append(card)

    return card_options


def render_public_player_stats(board: Board) -> str:
    longest_paths = find_longest_paths(board)
    data = defaultdict(list)
    for player_id in range(board.num_players):
        data["Player"].append(player_id)
        data["Total Visible Points"].append(
            board.route_points[player_id] + longest_paths.points[player_id]
        )
        data["Route Points"].append(board.route_points[player_id])
        data["Longest Path Points"].append(longest_paths.points[player_id])
        data["Longest Path Length"].append(longest_paths.lengths[player_id])
        data["Train Cars Left"].append(board.train_cars[player_id])

    return tabulate(data, headers="keys", tablefmt="grid")


def render_ticket(ticket: Ticket, connected: bool) -> str:
    points = ticket.value if connected else -ticket.value
    return f"  {ticket.source_city.name} --> {ticket.destination_city.name}: {points} points"


def print_player(board: Board, player: Player) -> None:
    print(colored("Cards in hand:", attrs=["bold"]))
    for color, num_cards in player.card_counts.items():
        print(f"  {color}: {num_cards} cards")

    connected = check_tickets(board, player)
    print(colored("Tickets:", attrs=["bold"]))
    for ticket, ticket_status in zip(player.tickets, connected):
        print(render_ticket(ticket, ticket_status))


def print_board(board: Board) -> None:
    print(colored("Visible cards: ", attrs=["bold"]) + render_cards(board.visible_cards))
    print(colored("Owned routes:", attrs=["bold"]))
    for route_info in board.route_ownership.values():
        print(f"  Player {route_info.player_id}: {route_info}")
    print(render_public_player_stats(board))


def print_state(board: Board, player: Player) -> None:
    print()
    print(colored("Board information:", color="red", attrs=["bold"]))
    print_board(board)
    print(colored("Player information:", color="red", attrs=["bold"]))
    print_player(board=board, player=player)
    print()
    print()


T = TypeVar("T")


def read_option(description: str, options: list[T]) -> T:
    print(description)
    for idx, action in enumerate(options):
        print(f"  {idx}. {action}")

    while True:
        option_index = input(
            f"Type a number between 0 and {len(options) - 1} to select an option: "
        )
        try:
            index = int(option_index)
            assert 0 <= index < len(options)
            print(f"Selected option {options[index]}")
            print()
            return options[index]
        except (ValueError, AssertionError) as e:
            print(f"Invalid option: {option_index}. Try again.")
