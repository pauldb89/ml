import collections
from collections import defaultdict
from dataclasses import asdict

from tabulate import tabulate
from termcolor import colored
from typing import TypeVar

from board_games.ticket2ride.board import Board
from board_games.ticket2ride.card import render_cards
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.longest_path import find_longest_paths
from board_games.ticket2ride.player import Player
from board_games.ticket2ride.route import ROUTES
from board_games.ticket2ride.state import ObservedState, PlayerScore
from board_games.ticket2ride.ticket import Ticket


def render_public_player_stats(board: Board) -> str:
    longest_paths = find_longest_paths(board)
    data = defaultdict(list)
    for player_id in range(board.num_players):
        data["player"].append(player_id)
        data["total_visible_points"].append(
            board.route_points[player_id] + longest_paths.points[player_id]
        )
        data["route_points"].append(board.route_points[player_id])
        data["longest_path_points"].append(longest_paths.points[player_id])
        data["longest_path"].append(longest_paths.lengths[player_id])
        data["train_cars_left"].append(board.train_cars[player_id])

    return tabulate(data, headers="keys", tablefmt="grid")


def render_ticket(ticket: Ticket, connected: bool) -> str:
    points = ticket.value if connected else -ticket.value
    return f"  {ticket.source_city.name} --> {ticket.destination_city.name}: {points} points"


def print_player(board: Board, player: Player) -> None:
    print(colored("Cards in hand:", attrs=["bold"]))
    for color, num_cards in player.card_counts.items():
        print(f"  {color}: {num_cards} cards")

    disjoint_sets = DisjointSets()
    for route_info in board.route_ownership.values():
        route = ROUTES[route_info.route_id]
        disjoint_sets.connect(route.source_city, route.destination_city)

    print(colored("Tickets:", attrs=["bold"]))
    for ticket in player.tickets:
        print(
            render_ticket(
                ticket,
                disjoint_sets.are_connected(ticket.source_city, ticket.destination_city),
            )
        )


def print_board(board: Board) -> None:
    print(colored("Visible cards: ", attrs=["bold"]) + render_cards(board.visible_cards))
    print(colored("Owned routes:", attrs=["bold"]))
    for route_info in board.route_ownership.values():
        print(f"  Player {route_info.player_id}: {route_info}")
    print(render_public_player_stats(board))


def print_state(state: ObservedState) -> None:
    print()
    print(colored("Board information:", color="red", attrs=["bold"]))
    print_board(state.board)
    print(colored("Player information:", color="red", attrs=["bold"]))
    print_player(board=state.board, player=state.player)
    print()
    print()


def print_scorecard(scorecard: list[PlayerScore]) -> None:
    print(colored("Scorecard:", color="blue", attrs=["bold"]))
    data = collections.defaultdict(list)
    for player_score in scorecard:
        for key, value in asdict(player_score).items():
            data[key].append(value)
        data["total_points"].append(player_score.total_points)

    print(tabulate(data, headers="keys", tablefmt="grid"))


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
        except (ValueError, AssertionError):
            print(f"Invalid option: {option_index}. Try again.")
