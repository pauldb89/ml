from typing import Callable

import pytest

from board_games.ticket2ride.data_models import (
    Board, Player, ANY, WHITE, RED, RouteInfo, get_build_route_options, BLUE, BLACK, DisjointSets,
    HELENA, SEATTLE, LOS_ANGELES, PHOENIX, DENVER
)


@pytest.fixture()
def assert_array_equal() -> Callable[[list, list], None]:
    def compare(arr1: list, arr2: list) -> None:
        assert sorted(arr1) == sorted(arr2), f"Arrays not equal: {sorted(arr1)} != {sorted(arr2)}"

    return compare


def test_get_build_route_options(
    assert_array_equal: Callable[[list, list], None],
) -> None:
    board = Board.init(num_players=2)
    player = Player(id=0, card_counts={ANY: 1, WHITE: 4, RED: 1})
    board.train_cars[player.id] = 3
    board.route_ownership = {
        1: RouteInfo(route_id=1, player_id=0, color=BLUE, num_any_cards=0),
        2: RouteInfo(route_id=2, player_id=1, color=BLACK, num_any_cards=0),
    }
    options: list[RouteInfo] = get_build_route_options(board=board, player=player)

    expected_options = [
        # WHITE routes.
        RouteInfo(route_id=35, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=42, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=78, player_id=0, color=WHITE, num_any_cards=0),
        # RED routes.
        RouteInfo(route_id=22, player_id=0, color=RED, num_any_cards=1),
        # GRAY routes with just GRAY cards.
        RouteInfo(route_id=15, player_id=0, color=ANY, num_any_cards=1),
        RouteInfo(route_id=16, player_id=0, color=ANY, num_any_cards=1),
        RouteInfo(route_id=47, player_id=0, color=ANY, num_any_cards=1),
        RouteInfo(route_id=48, player_id=0, color=ANY, num_any_cards=1),
        RouteInfo(route_id=66, player_id=0, color=ANY, num_any_cards=1),
        RouteInfo(route_id=90, player_id=0, color=ANY, num_any_cards=1),
        RouteInfo(route_id=91, player_id=0, color=ANY, num_any_cards=1),
        # GRAY routes with RED (and maybe some ANY cards).
        RouteInfo(route_id=15, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=16, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=47, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=48, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=66, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=90, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=91, player_id=0, color=RED, num_any_cards=0),
        RouteInfo(route_id=9, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=12, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=13, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=18, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=32, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=33, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=46, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=50, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=51, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=56, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=59, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=60, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=63, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=64, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=67, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=68, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=71, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=72, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=74, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=75, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=76, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=77, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=80, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=92, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=95, player_id=0, color=RED, num_any_cards=1),
        RouteInfo(route_id=99, player_id=0, color=RED, num_any_cards=1),
        # GRAY routes with WHITE.
        RouteInfo(route_id=0, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=9, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=10, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=11, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=12, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=13, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=15, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=16, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=18, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=32, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=33, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=46, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=47, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=48, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=50, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=51, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=56, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=58, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=59, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=60, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=63, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=64, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=66, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=67, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=68, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=71, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=72, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=74, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=75, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=76, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=77, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=80, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=85, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=87, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=90, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=91, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=92, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=95, player_id=0, color=WHITE, num_any_cards=0),
        RouteInfo(route_id=99, player_id=0, color=WHITE, num_any_cards=0),
    ]

    assert_array_equal(expected_options, options)


def test_disjoint_sets() -> None:
    disjoint_sets = DisjointSets()

    assert not disjoint_sets.are_connected(SEATTLE, HELENA)

    disjoint_sets.connect(SEATTLE, HELENA)
    assert disjoint_sets.are_connected(SEATTLE, HELENA)

    assert not disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
    disjoint_sets.connect(LOS_ANGELES, PHOENIX)
    assert not disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
    disjoint_sets.connect(HELENA, DENVER)
    assert not disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
    disjoint_sets.connect(DENVER, PHOENIX)
    assert disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
