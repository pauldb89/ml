from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, order=True)
class Color:
    id: int
    name: str

    def __hash__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return f"{self.name}"


@dataclass(frozen=True)
class City:
    id: int
    name: str


@dataclass(frozen=True)
class Card:
    color: Color

    def __repr__(self) -> str:
        return f"{self.color.name}"

    def __hash__(self) -> int:
        return hash(self.color)


ROUTE_LENGTHS_TO_VALUES: dict[int, int] = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    5: 10,
    6: 15,
}


@dataclass(frozen=True)
class Route:
    id: int
    source_city: City
    destination_city: City
    color: Color
    length: int

    @property
    def value(self) -> int:
        if self.length not in ROUTE_LENGTHS_TO_VALUES:
            raise ValueError("Invalid route length")

        return ROUTE_LENGTHS_TO_VALUES[self.length]


@dataclass(order=True)
class RouteInfo:
    route_id: int
    player_id: int
    color: Color
    num_any_cards: int


@dataclass(frozen=True)
class Ticket:
    id: int
    source_city: City
    destination_city: City
    value: int


class ActionType(StrEnum):
    DRAW_CARDS = "DRAW_CARDS"
    BUILD_ROUTE = "BUILD_ROUTE"
    DRAW_TICKETS = "DRAW_TICKETS"


@dataclass(frozen=True)
class Action:
    player_id: int
    action_type: ActionType


@dataclass(frozen=True)
class DrawCards(Action):
    cards: list[Card | None]


@dataclass(frozen=True)
class DrawTickets(Action):
    tickets: list[Ticket]


@dataclass(frozen=True)
class BuildRoute(Action):
    route_info: RouteInfo
