import copy
import random
from board_games.ticket2ride.ticket import TICKETS, DrawnTickets, Ticket


class TicketDeck:
    tickets: list[Ticket]

    def __init__(self, rng: random.Random) -> None:
        self.tickets = copy.deepcopy(TICKETS)
        rng.shuffle(self.tickets)

    def get(self) -> DrawnTickets:
        assert len(self.tickets) >= 3
        return self.tickets.pop(), self.tickets.pop(), self.tickets.pop()

    def __len__(self) -> int:
        return len(self.tickets)
