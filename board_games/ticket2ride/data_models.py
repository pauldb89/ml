from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, order=True)
class Color:
    id: int
    name: str


@dataclass(frozen=True)
class City:
    id: int
    name: str


@dataclass(frozen=True)
class Card:
    color: Color


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
