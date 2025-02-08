from dataclasses import dataclass

from board_games.ticket2ride.city import (
    SALT_LAKE_CITY,
    City,
    ATLANTA,
    BOSTON,
    CALGARY,
    CHICAGO,
    DALLAS,
    DENVER,
    DULUTH,
    EL_PASO,
    HELENA,
    HOUSTON,
    KANSAS_CITY,
    LITTLE_ROCK,
    LOS_ANGELES,
    MIAMI,
    MONTREAL,
    NASHVILLE,
    NEW_ORLEANS,
    NEW_YORK,
    OKLAHOMA_CITY,
    PHOENIX,
    PITTSBURGH,
    PORTLAND,
    SANTA_FE,
    SAULT_ST_MARIE,
    SEATTLE,
    SAN_FRANCISCO,
    TORONTO,
    VANCOUVER,
    WINNIPEG,
)

@dataclass(frozen=True, order=True)
class Ticket:
    id: int
    source_city: City
    destination_city: City
    value: int

    def __repr__(self) -> str:
        return f"{self.source_city.name} - {self.destination_city.name} ({self.value} points)"


Tickets = tuple[Ticket, ...]
DrawnTickets = tuple[Ticket, Ticket, Ticket]


TICKETS: list[Ticket] = [
    Ticket(id=0, source_city=DENVER, destination_city=EL_PASO, value=4),
    Ticket(id=1, source_city=KANSAS_CITY, destination_city=HOUSTON, value=5),
    Ticket(id=2, source_city=NEW_YORK, destination_city=ATLANTA, value=6),
    Ticket(id=3, source_city=CALGARY, destination_city=SALT_LAKE_CITY, value=7),
    Ticket(id=4, source_city=CHICAGO, destination_city=NEW_ORLEANS, value=7),
    Ticket(id=5, source_city=HELENA, destination_city=LOS_ANGELES, value=8),
    Ticket(id=6, source_city=DULUTH, destination_city=HOUSTON, value=8),
    Ticket(id=7, source_city=SAULT_ST_MARIE, destination_city=NASHVILLE, value=8),
    Ticket(id=8, source_city=SEATTLE, destination_city=LOS_ANGELES, value=9),
    Ticket(id=9, source_city=MONTREAL, destination_city=ATLANTA, value=9),
    Ticket(id=10, source_city=CHICAGO, destination_city=SANTA_FE, value=9),
    Ticket(id=11, source_city=SAULT_ST_MARIE, destination_city=OKLAHOMA_CITY, value=9),
    Ticket(id=12, source_city=TORONTO, destination_city=MIAMI, value=10),
    Ticket(id=13, source_city=DULUTH, destination_city=EL_PASO, value=10),
    Ticket(id=14, source_city=PORTLAND, destination_city=PHOENIX, value=11),
    Ticket(id=15, source_city=DALLAS, destination_city=NEW_YORK, value=11),
    Ticket(id=16, source_city=DENVER, destination_city=PITTSBURGH, value=11),
    Ticket(id=17, source_city=WINNIPEG, destination_city=LITTLE_ROCK, value=11),
    Ticket(id=18, source_city=WINNIPEG, destination_city=HOUSTON, value=12),
    Ticket(id=19, source_city=BOSTON, destination_city=MIAMI, value=12),
    Ticket(id=20, source_city=CALGARY, destination_city=PHOENIX, value=13),
    Ticket(id=21, source_city=MONTREAL, destination_city=NEW_ORLEANS, value=13),
    Ticket(id=22, source_city=VANCOUVER, destination_city=SANTA_FE, value=13),
    Ticket(id=23, source_city=LOS_ANGELES, destination_city=CHICAGO, value=16),
    Ticket(id=24, source_city=PORTLAND, destination_city=NASHVILLE, value=17),
    Ticket(id=25, source_city=SAN_FRANCISCO, destination_city=ATLANTA, value=17),
    Ticket(id=26, source_city=VANCOUVER, destination_city=MONTREAL, value=20),
    Ticket(id=27, source_city=LOS_ANGELES, destination_city=MIAMI, value=20),
    Ticket(id=28, source_city=LOS_ANGELES, destination_city=NEW_YORK, value=21),
    Ticket(id=29, source_city=SEATTLE, destination_city=NEW_YORK, value=22),
]
