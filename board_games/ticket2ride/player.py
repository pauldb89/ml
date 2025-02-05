from collections import defaultdict
from board_games.ticket2ride.color import Color
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.route import Route
from board_games.ticket2ride.ticket import Ticket, Tickets


class Player:
    id: int
    card_counts: dict[Color, int]
    tickets: list[Ticket]
    disjoint_sets: DisjointSets

    def __init__(
        self,
        player_id: int,
        card_counts: dict[Color, int] | None = None,
        tickets: Tickets | None = None,
    ) -> None:
        self.id = player_id
        self.card_counts = card_counts or defaultdict(int)
        self.tickets = tickets or []
        self.disjoint_sets = DisjointSets()

    def build_route(self, route: Route) -> None:
        self.disjoint_sets.connect(route.source_city, route.destination_city)
