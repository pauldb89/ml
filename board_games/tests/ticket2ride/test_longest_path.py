from board_games.ticket2ride.longest_path import find_longest_path
from board_games.ticket2ride.entities import ROUTES


def test_longest_path() -> None:
    routes = [
        ROUTES[94],
        ROUTES[4],
        ROUTES[17],
        ROUTES[6],
        ROUTES[28],
        ROUTES[1],
        ROUTES[34],
        ROUTES[7],
        ROUTES[19],
    ]
    assert find_longest_path(routes) == 37
