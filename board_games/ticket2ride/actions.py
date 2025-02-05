from dataclasses import dataclass
from enum import StrEnum

from board_games.ticket2ride.card import Card
from board_games.ticket2ride.route_info import RouteInfo
from board_games.ticket2ride.ticket import Tickets


class ActionType(StrEnum):
    PLAN = "PLAN"
    DRAW_CARD = "DRAW_CARD"
    BUILD_ROUTE = "BUILD_ROUTE"
    DRAW_TICKETS = "DRAW_TICKETS"


@dataclass(frozen=True)
class Prediction:
    class_id: int
    log_prob: float
    value: float


@dataclass(frozen=True)
class Action:
    player_id: int
    action_type: ActionType
    prediction: Prediction | None


@dataclass(frozen=True)
class Plan(Action):
    next_action_type: ActionType

    def __repr__(self) -> str:
        return f"Player {self.player_id} decided to take action {self.next_action_type.value}"


@dataclass(frozen=True)
class DrawCard(Action):
    card: Card | None

    def __repr__(self) -> str:
        if self.card is None:
            return f"Player {self.player_id} drew card from deck"
        else:
            return f"Player {self.player_id} drew card {self.card}"


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
