import copy
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Self

from board_games.ticket2ride.actions import ActionType, Action
from board_games.ticket2ride.board import Board
from board_games.ticket2ride.consts import LONGEST_PATH_POINTS
from board_games.ticket2ride.player import Player
from board_games.ticket2ride.ticket import DrawnTickets


class ObservedState:
    board: Board
    player: Player
    next_action: ActionType
    drawn_tickets: DrawnTickets | None

    turn_id: int
    terminal: bool
    consecutive_card_draws: int

    def __init__(
        self,
        board: Board,
        player: Player,
        action_type: ActionType,
        terminal: bool = False,
        turn_id: int = 0,
        consecutive_card_draws: int = 0,
        drawn_tickets: DrawnTickets | None = None,
    ) -> None:
        self.board = copy.deepcopy(board)
        self.player = copy.deepcopy(player)
        self.next_action = action_type
        self.drawn_tickets = drawn_tickets

        self.turn_id = turn_id
        self.terminal = terminal
        self.consecutive_card_draws = consecutive_card_draws


@dataclass
class PlayerScore:
    player_id: int
    route_points: int = 0
    ticket_points: int = 0
    total_tickets: int = 0
    completed_tickets: int = 0
    longest_path_bonus: bool = False
    longest_path: int = 0
    owned_routes_by_length: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total_points(self) -> int:
        return (
            self.route_points
            + self.ticket_points
            + (LONGEST_PATH_POINTS if self.longest_path_bonus else 0)
        )

    @property
    def sort_weight(self) -> tuple[int, int, int]:
        return self.total_points, self.completed_tickets, self.longest_path

    def __lt__(self, other: Self) -> bool:
        return self.sort_weight < other.sort_weight


@dataclass
class Score:
    scorecard: list[PlayerScore]
    turn_score: PlayerScore

    @cached_property
    def winner_id(self) -> int:
        winner_id = 0
        for player_id, player_score in enumerate(self.scorecard):
            if player_score > self.scorecard[winner_id]:
                winner_id = player_id
        return winner_id


@dataclass
class Transition:
    action: Action
    state: ObservedState
    score: Score
