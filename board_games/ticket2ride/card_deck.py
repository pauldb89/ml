from dataclasses import dataclass
import random
from board_games.ticket2ride.card import Card
from board_games.ticket2ride.color import ANY, COLORS
from board_games.ticket2ride.consts import NUM_ANY_CARDS, NUM_COLOR_CARDS


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