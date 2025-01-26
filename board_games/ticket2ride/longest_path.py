from dataclasses import dataclass

from board_games.ticket2ride.board_logic import Board
from board_games.ticket2ride.consts import LONGEST_PATH_POINTS
from board_games.ticket2ride.entities import Route, CITIES, ROUTES


@dataclass
class LongestPaths:
    lengths: list[int]
    points: list[int]


def _find_longest_path(
    graph: list[list[tuple[int, int, int]]],
    node: int,
    visited_routes: set[int]
) -> int:
    max_length = 0
    for neighbor, route_id, length in graph[node]:
        if route_id not in visited_routes:
            visited_routes.add(route_id)
            max_length = max(
                max_length, _find_longest_path(graph, neighbor, visited_routes) + length
            )
            visited_routes.remove(route_id)

    return max_length


def find_longest_path(routes: list[Route]) -> int:
    graph = [[] for _ in range(len(CITIES))]
    for route in routes:
        graph[route.source_city.id].append((route.destination_city.id, route.id, route.length))
        graph[route.destination_city.id].append((route.source_city.id, route.id, route.length))

    max_length = 0
    for node in range(len(CITIES)):
        max_length = max(max_length, _find_longest_path(graph, node, visited_routes=set()))

    return max_length


def find_longest_paths(board: Board) -> LongestPaths:
    longest_paths = []
    for player_id in range(board.num_players):
        routes = []
        for route_info in board.route_ownership.values():
            if route_info.player_id == player_id:
                routes.append(ROUTES[route_info.route_id])

        longest_paths.append(find_longest_path(routes))

    max_path_length = max(longest_paths)
    bonus_points = []
    for path_length in longest_paths:
        bonus_points.append(LONGEST_PATH_POINTS if max_path_length == path_length else 0)

    return LongestPaths(lengths=longest_paths, points=bonus_points)
