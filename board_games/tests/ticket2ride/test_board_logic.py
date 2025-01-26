from typing import Callable

import pytest

from board_games.ticket2ride.board_logic import Board, Player, count_ticket_points, RouteInfo
from board_games.ticket2ride.entities import ANY, WHITE, RED, BLUE, BLACK, GREEN, TICKETS


@pytest.fixture()
def assert_array_equal() -> Callable[[list, list], None]:
    def compare(arr1: list, arr2: list) -> None:
        assert sorted(arr1) == sorted(arr2), f"Arrays not equal: {sorted(arr1)} != {sorted(arr2)}"

    return compare


def test_count_ticket_points() -> None:
    board = Board(num_players=3)
    board.route_ownership = {
        0: RouteInfo(route_id=0, player_id=0, color=RED, num_any_cards=0),
        8: RouteInfo(route_id=8, player_id=0, color=BLACK, num_any_cards=0),
        3: RouteInfo(route_id=3, player_id=0, color=WHITE, num_any_cards=0),
        5: RouteInfo(route_id=5, player_id=0, color=GREEN, num_any_cards=0),
        2: RouteInfo(route_id=2, player_id=0, color=ANY, num_any_cards=1),
        14: RouteInfo(route_id=14, player_id=0, color=BLUE, num_any_cards=1),
        56: RouteInfo(route_id=56, player_id=1, color=BLUE, num_any_cards=0),
        72: RouteInfo(route_id=72, player_id=0, color=BLACK, num_any_cards=1),
    }
    player1 = Player(player_id=0, tickets=(TICKETS[25], TICKETS[26], TICKETS[29]))
    player2 = Player(player_id=1, tickets=(TICKETS[0], TICKETS[15]))
    player3 = Player(player_id=2, tickets=(TICKETS[9],))

    assert count_ticket_points(board=board, player=player1) == 25
    assert count_ticket_points(board=board, player=player2) == -7
    assert count_ticket_points(board=board, player=player3) == -9

