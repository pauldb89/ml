from board_games.ticket2ride import consts
from board_games.ticket2ride.card import Card
from board_games.ticket2ride.card_deck import CardDeck
from board_games.ticket2ride.color import ANY
from board_games.ticket2ride.consts import MAX_VISIBLE_ANY_CARDS, NUM_INITIAL_TRAIN_CARS, NUM_VISIBLE_CARDS
from board_games.ticket2ride.route_info import RouteInfo
from board_games.ticket2ride.ticket_deck import TicketDeck


class InvalidGameStateError(Exception):
    pass


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
        while True:
            while len(self.card_deck) > 0 and len(self.visible_cards) < NUM_VISIBLE_CARDS:
                self.visible_cards.append(self.card_deck.draw())

            num_visible_any_cards = 0
            for card in self.visible_cards:
                if card.color == ANY:
                    num_visible_any_cards += 1

            if num_visible_any_cards > MAX_VISIBLE_ANY_CARDS:
                remaining_regular_cards = (
                    self.card_deck.remaining_regular_cards
                    + len(self.visible_cards) - num_visible_any_cards
                )
                if remaining_regular_cards < consts.NUM_VISIBLE_CARDS - consts.MAX_VISIBLE_ANY_CARDS:
                    raise InvalidGameStateError("Insufficient regular cards left")

                for card in self.visible_cards:
                    self.card_deck.discard(card)
                self.visible_cards = []
            else:
                break
