from typing import Callable, Protocol, Self

import pytest

from board_games.ticket2ride.actions import ActionType
from board_games.ticket2ride.action_utils import (
    get_build_route_options,
    get_draw_card_options,
    get_ticket_draw_options,
    get_valid_actions,
)
from board_games.ticket2ride.board import Board
from board_games.ticket2ride.card import Card
from board_games.ticket2ride.color import ANY, BLACK, BLUE, RED, WHITE
from board_games.ticket2ride.player import Player
from board_games.ticket2ride.route_info import RouteInfo
from board_games.ticket2ride.state import ObservedState
from board_games.ticket2ride.ticket import TICKETS


class Comparable(Protocol):
    def __lt__(self, other: Self) -> bool:
        ...


@pytest.fixture()
def assert_item_count_equal() -> Callable[[list[Comparable | None], list[Comparable | None]], None]:
    def compare(arr1: list[Comparable | None], arr2: list[Comparable | None]) -> None:
        sorted_arr1 = sorted(arr1, key=lambda x: (x is None, x))
        sorted_arr2 = sorted(arr2, key=lambda x: (x is None, x))
        assert sorted_arr1 == sorted_arr2, f"Arrays not equal: {sorted_arr1} != {sorted_arr2}"

    return compare


def test_get_valid_actions(assert_item_count_equal: Callable[[list, list], None]) -> None:
    board = Board(num_players=2)
    assert len(board.visible_cards) == 5
    assert len(board.ticket_deck) == 30

    state = ObservedState(
        board=board,
        player=Player(player_id=0, card_counts={ANY: 1}),
        action_type=ActionType.PLAN,
        turn_id=0,
    )

    assert_item_count_equal(get_valid_actions(state), [ActionType.DRAW_TICKETS])

    state.turn_id = 1

    assert_item_count_equal(
        get_valid_actions(state),
        [ActionType.DRAW_CARD, ActionType.DRAW_TICKETS, ActionType.BUILD_ROUTE],
    )

    state.player.card_counts = {}
    assert_item_count_equal(
        get_valid_actions(state),
        [ActionType.DRAW_CARD, ActionType.DRAW_TICKETS],
    )

    state.board.ticket_deck.tickets = []
    assert_item_count_equal(get_valid_actions(state), [ActionType.DRAW_CARD])

    state.board.visible_cards = []
    state.board.card_deck.deck = [Card(color=ANY)]
    assert_item_count_equal(get_valid_actions(state), [])


def test_get_draw_card_options(assert_item_count_equal: Callable[[list, list], None]) -> None:
    board = Board(num_players=2)
    board.visible_cards = [
        Card(color=WHITE), Card(color=BLACK), Card(color=WHITE), Card(color=ANY), Card(color=BLACK)
    ]

    state = ObservedState(board=board, player=Player(player_id=0), action_type=ActionType.PLAN)

    state.consecutive_card_draws = 0
    draw_options = get_draw_card_options(state)
    expected_options = [Card(color=WHITE), Card(color=BLACK), Card(color=ANY), None]
    assert_item_count_equal(draw_options, expected_options)

    state.consecutive_card_draws = 1
    draw_options = get_draw_card_options(state)
    expected_options = [Card(color=WHITE), Card(color=BLACK), None]
    assert_item_count_equal(draw_options, expected_options)

    state.consecutive_card_draws = 0
    state.board.card_deck.deck = []
    draw_options = get_draw_card_options(state)
    expected_options = [Card(color=WHITE), Card(color=BLACK), Card(color=ANY)]
    assert_item_count_equal(draw_options, expected_options)

    state.consecutive_card_draws = 1
    draw_options = get_draw_card_options(state)
    expected_options = [Card(color=WHITE), Card(color=BLACK)]
    assert_item_count_equal(draw_options, expected_options)


def test_get_build_route_options(assert_item_count_equal: Callable[[list, list], None]) -> None:
    board = Board(num_players=2)
    player = Player(player_id=0, card_counts={ANY: 1, WHITE: 4, RED: 1})
    board.train_cars[player.id] = 3
    board.route_ownership = {
        1: RouteInfo(route_id=1, player_id=0, color=BLUE, num_any_cards=0),
        2: RouteInfo(route_id=2, player_id=1, color=BLACK, num_any_cards=0),
    }
    options: list[RouteInfo] = get_build_route_options(
        ObservedState(board=board, player=player, action_type=ActionType.BUILD_ROUTE)
    )

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

    assert_item_count_equal(expected_options, options)


def test_get_ticket_draw_options(assert_item_count_equal: Callable[[list, list], None]) -> None:
    drawn_tickets = (TICKETS[0], TICKETS[1], TICKETS[2])
    ticket_options = get_ticket_draw_options(drawn_tickets, is_initial_turn=True)
    expected_options = [
        (TICKETS[0], TICKETS[1], TICKETS[2]),
        (TICKETS[0], TICKETS[1]),
        (TICKETS[0], TICKETS[2]),
        (TICKETS[1], TICKETS[2]),
    ]
    assert_item_count_equal(ticket_options, expected_options)

    ticket_options = get_ticket_draw_options(drawn_tickets, is_initial_turn=False)
    expected_options = [
        (TICKETS[0], TICKETS[1], TICKETS[2]),
        (TICKETS[0], TICKETS[1]),
        (TICKETS[0], TICKETS[2]),
        (TICKETS[1], TICKETS[2]),
        (TICKETS[0],),
        (TICKETS[1],),
        (TICKETS[2],),
    ]
    assert_item_count_equal(ticket_options, expected_options)
