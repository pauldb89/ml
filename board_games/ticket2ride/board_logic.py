import copy
import random
from collections import defaultdict
from dataclasses import dataclass

from board_games.ticket2ride import consts
from board_games.ticket2ride.consts import MAX_VISIBLE_ANY_CARDS, ANY, NUM_VISIBLE_CARDS, \
    NUM_INITIAL_TRAIN_CARS, COLORS, NUM_COLOR_CARDS, NUM_ANY_CARDS, TICKETS, ROUTES
from board_games.ticket2ride.data_models import Color, Ticket, Card, DrawnTickets, \
    Tickets
from board_games.ticket2ride.disjoint_sets import DisjointSets


@dataclass(order=True)
class RouteInfo:
    route_id: int
    player_id: int
    color: Color
    num_any_cards: int

    def __repr__(self) -> str:
        route = ROUTES[self.route_id]
        return f"{route} with {self.color} cards and {self.num_any_cards} additional {ANY} cards"


class InvalidGameStateError(Exception):
    pass


class TicketDeck:
    tickets: list[Ticket]

    def __init__(self) -> None:
        self.tickets = copy.deepcopy(TICKETS)
        random.shuffle(self.tickets)

    def get(self) -> DrawnTickets:
        assert len(self.tickets) >= 3
        return tuple(self.tickets.pop() for _ in range(3))

    def __len__(self) -> int:
        return len(self.tickets)


@dataclass
class CardDeck:
    deck: list[Card]
    discard_pile: list[Card]

    def __init__(self) -> None:
        self.discard_pile = []
        self.deck = [Card(color=ANY) for _ in range(NUM_ANY_CARDS)]
        for color in COLORS:
            for _ in range(NUM_COLOR_CARDS):
                self.deck.append(Card(color=color))

        random.shuffle(self.deck)

    def draw(self) -> Card:
        if len(self.deck) == 0:
            self.deck = self.discard_pile
            self.discard_pile = []
            random.shuffle(self.deck)

        assert len(self.deck) > 0
        return self.deck.pop()

    def discard(self, card: Card) -> None:
        self.discard_pile.append(card)

    @property
    def remaining_regular_cards(self) -> int:
        return len([card for card in self.deck + self.discard_pile if card.color != ANY])

    def __len__(self) -> int:
        return len(self.deck) + len(self.discard_pile)

    def __str__(self) -> str:
        return f"CardDeck(deck={self.deck}, discard_pile={self.discard_pile})"


class Board:
    route_ownership: dict[int, RouteInfo]
    ticket_deck: TicketDeck
    card_deck: CardDeck
    train_cars: list[int]
    route_points: list[int]
    visible_cards: list[Card]
    num_players: int

    def __init__(self, num_players: int) -> None:
        self.num_players = num_players

        self.route_ownership = {}
        self.ticket_deck = TicketDeck()
        self.card_deck = CardDeck()

        self.train_cars = [NUM_INITIAL_TRAIN_CARS for _ in range(num_players)]
        self.route_points = [0 for _ in range(num_players)]

        self.visible_cards = []
        self.reveal_cards()

    def reveal_cards(self) -> None:
        remaining_regular_cards = self.card_deck.remaining_regular_cards
        for card in self.visible_cards:
            if card.color != ANY:
                remaining_regular_cards += 1

        if remaining_regular_cards < consts.NUM_VISIBLE_CARDS - consts.MAX_VISIBLE_ANY_CARDS:
            raise InvalidGameStateError("Insufficient regular cards left")

        while True:
            while len(self.card_deck) > 0 and len(self.visible_cards) < NUM_VISIBLE_CARDS:
                self.visible_cards.append(self.card_deck.draw())

            num_visible_any_cards = 0
            for card in self.visible_cards:
                if card.color == ANY:
                    num_visible_any_cards += 1

            if num_visible_any_cards > MAX_VISIBLE_ANY_CARDS:
                for card in self.visible_cards:
                    self.card_deck.discard(card)
                self.visible_cards = []
            else:
                break


class Player:
    id: int
    card_counts: dict[Color, int]
    tickets: list[Ticket]

    def __init__(
        self,
        player_id: int,
        card_counts: dict[Color, int] | None = None,
        tickets: Tickets | None = None,
    ) -> None:
        self.id = player_id
        self.card_counts = card_counts or defaultdict(int)
        self.tickets = tickets or []


def check_tickets(board: Board, player: Player) -> list[bool]:
    disjoint_sets = DisjointSets()
    for route_info in board.route_ownership.values():
        route = ROUTES[route_info.route_id]
        disjoint_sets.connect(route.source_city, route.destination_city)

    connected = []
    for ticket in player.tickets:
        connected.append(disjoint_sets.are_connected(ticket.source_city, ticket.destination_city))
    return connected


def count_ticket_points(board: Board, player: Player) -> int:
    points = 0
    connected = check_tickets(board, player)
    for ticket, ticket_status in zip(player.tickets, connected):
        points += ticket.value if ticket_status else -ticket.value
    return points
