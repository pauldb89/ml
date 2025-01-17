import itertools
from collections import defaultdict
from typing import TypeVar

from tabulate import tabulate
from termcolor import colored

from board_games.ticket2ride.board_logic import Board, Player, check_tickets
from board_games.ticket2ride.consts import ROUTES, ANY, COLORS, RED, WHITE, BLUE, YELLOW, \
    ORANGE, BLACK, GREEN, PINK
from board_games.ticket2ride.data_models import RouteInfo, ActionType, Color, Card, Ticket
from board_games.ticket2ride.longest_path import find_longest_paths


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
def get_ticket_draw_options(tickets: list[Ticket], is_initial_turn: bool) -> list[list[Ticket]]:
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


def render_color(color: Color) -> str:
    color_map: dict[Color, str] = {
        PINK: "light_magenta",
        WHITE: "white",
        BLUE: "blue",
        YELLOW: "yellow",
        ORANGE: "light_red",
        BLACK: "black",
        RED: "red",
        GREEN: "green",
        ANY: "magenta",
    }
    return colored(color.name, color=color_map[color])


def render_card(card: Card) -> str:
    return render_color(card.color)


def render_visible_cards(board: Board) -> str:
    return ", ".join([render_card(card) for card in board.visible_cards])


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
    print("Cards in hand:")
    for color, num_cards in player.card_counts.items():
        print(f"  {render_color(color)}: {num_cards} cards")

    connected = check_tickets(board, player)
    print("Tickets:")
    for ticket, ticket_status in zip(player.tickets, connected):
        print(render_ticket(ticket, ticket_status))


def print_board(board: Board) -> None:
    print("*" * 20)
    print(f"Visible cards: {render_visible_cards(board)}")
    print(render_public_player_stats(board))


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
            return options[index]
        except (ValueError, AssertionError) as e:
            print(f"Invalid option: {option_index}. Try again.")
