from board_games.ticket2ride.consts import CITIES
from board_games.ticket2ride.data_models import Route


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
