from dataclasses import dataclass

from termcolor import colored


@dataclass(frozen=True, order=True)
class Color:
    id: int
    name: str

    def __repr__(self) -> str:
        color_overrides: dict[str, str] = {
            "pink": "light_magenta",
            "orange": "light_red",
            "any": "magenta",
        }
        return f"{colored(self.name, color=color_overrides.get(self.name, self.name))}"


@dataclass(frozen=True)
class City:
    id: int
    name: str


@dataclass(frozen=True)
class Card:
    color: Color

    def __repr__(self) -> str:
        return f"{self.color}"


def render_cards(cards: list[Card]) -> str:
    return ", ".join([repr(card) for card in cards])


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

    def __repr__(self) -> str:
        return (
            f"{self.source_city.name} - {self.destination_city.name} ({self.color}, {self.length})"
        )


@dataclass(frozen=True)
class Ticket:
    id: int
    source_city: City
    destination_city: City
    value: int

    def __repr__(self) -> str:
        return f"{self.source_city.name} - {self.destination_city.name} ({self.value} points)"


Tickets = tuple[Ticket, ...]
DrawnTickets = tuple[Ticket | Ticket | Ticket]


