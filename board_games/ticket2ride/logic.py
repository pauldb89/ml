import abc
import copy
import itertools
import random
from dataclasses import dataclass, Field, field
from typing import Self

from board_games.ticket2ride.consts import NUM_LAST_TURN_CARS, NUM_INITIAL_PLAYER_CARDS, ANY, \
    ROUTES, NUM_VISIBLE_CARDS, NUM_INITIAL_TRAIN_CARS, COLORS, NUM_COLOR_CARDS, NUM_ANY_CARDS, \
    TICKETS
from board_games.ticket2ride.data_models import Card, Ticket, Color, ActionType
from board_games.ticket2ride.disjoint_sets import DisjointSets


@dataclass(order=True)
class RouteInfo:
    route_id: int
    player_id: int
    color: Color
    num_any_cards: int


class TicketDeck:
    tickets: list[Ticket]

    def __init__(self) -> None:
        self.tickets = copy.deepcopy(TICKETS)
        random.shuffle(self.tickets)

    def get(self) -> list[Ticket]:
        assert len(self.tickets) >= 3
        return [self.tickets.pop() for _ in range(3)]

    def __len__(self) -> int:
        return len(self.tickets)


@dataclass
class DiscardPile:
    cards: list[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        self.cards.append(card)

    def __len__(self) -> int:
        return len(self.cards)

    def reset(self) -> None:
        self.cards = []


@dataclass
class Deck:
    cards: list[Card]

    @classmethod
    def init(cls) -> Self:
        cards = [Card(color=ANY) for _ in range(NUM_ANY_CARDS)]
        for color in COLORS:
            for _ in range(NUM_COLOR_CARDS):
                cards.append(Card(color=color))

        random.shuffle(cards)
        return cls(cards=cards)

    @classmethod
    def from_discard_pile(cls, pile: DiscardPile) -> Self:
        cards = pile.cards
        random.shuffle(cards)
        pile.reset()
        return cls(cards=cards)

    def __len__(self) -> int:
        return len(self.cards)

    def get(self) -> Card:
        assert len(self.cards) > 0
        return self.cards.pop()


class Board:
    route_ownership: dict[int, RouteInfo]
    ticket_deck: TicketDeck
    card_deck: Deck
    discard_pile: DiscardPile
    visible_cards: list[Card]
    train_cars: list[int]

    # TODO(pauldb): Keep track of visible points.
    def __init__(self, num_players: int) -> None:
        self.route_ownership = {}
        self.ticket_deck = TicketDeck()
        self.card_deck = Deck.init()
        self.discard_pile = DiscardPile()

        # TODO(pauldb): Handle edge case where the initial cards have 3 or more locomotive cards.
        self.visible_cards = [self.card_deck.get() for _ in range(NUM_VISIBLE_CARDS)]
        self.train_cars = [NUM_INITIAL_TRAIN_CARS for _ in range(num_players)]


# TODO(pauldb): Maybe rename to PlayerData or PlayerInfo?
class Player:
    id: int
    card_counts: dict[Color, int]
    tickets: list[Ticket]

    # TODO(pauldb): Keep track of points including hidden information known by this user.
    def __init__(
        self,
        player_id: int,
        card_counts: dict[Color, int] | None = None,
        tickets: list[Ticket] | None = None,
    ) -> None:
        self.id = player_id
        self.card_counts = card_counts or {}
        self.tickets = tickets or []


def get_build_route_options(board: Board, player: Player) -> list[RouteInfo]:
    route_options: list[RouteInfo] = []
    for route in ROUTES:
        # This route has already been built, so it's not a valid option.
        if route.id in board.route_ownership:
            continue

        # The route is too long for the number of train cars the player currently has left.
        if route.length > board.train_cars[player.id]:
            continue

        # Check if we can use locomotive cards alone to build the route.
        if ANY in player.card_counts and player.card_counts[ANY] >= route.length:
            route_options.append(
                RouteInfo(
                    route_id=route.id,
                    player_id=player.id,
                    color=ANY,
                    num_any_cards=route.length,
                )
            )

        color_options = COLORS if route.color == ANY else [route.color]
        for color in color_options:
            if (
                    color in player.card_counts
                    and player.card_counts[color] + player.card_counts.get(ANY, 0) >= route.length
            ):
                route_options.append(
                    RouteInfo(
                        route_id=route.id,
                        player_id=player.id,
                        color=color,
                        # Greedily first use cards of the given color, then use locomotives (ANY).
                        num_any_cards=max(0, route.length - player.card_counts[color]),
                    )
                )

    return route_options


def count_ticket_points(board: Board, player: Player) -> int:
    disjoint_sets = DisjointSets()
    for route_info in board.route_ownership.values():
        route = ROUTES[route_info.route_id]
        disjoint_sets.connect(route.source_city, route.destination_city)

    points = 0
    for ticket in player.tickets:
        if disjoint_sets.are_connected(ticket.source_city, ticket.destination_city):
            points += ticket.value
        else:
            points -= ticket.value

    return points


def get_valid_actions(board: Board, player: Player) -> list[ActionType]:
    valid_action_types = []
    if len(board.card_deck) + len(board.discard_pile) >= 2:
        valid_action_types.append(ActionType.DRAW_CARDS)

    if len(board.ticket_deck) >= 3:
        valid_action_types.append(ActionType.DRAW_TICKETS)

    build_route_options = get_build_route_options(board, player)
    if len(build_route_options) > 0:
        valid_action_types.append(ActionType.BUILD_ROUTE)

    return valid_action_types


class Policy:
    @abc.abstractmethod
    def choose_action(self, board: Board, player: Player) -> ActionType:
        pass

    @abc.abstractmethod
    def choose_tickets(
        self,
        board: Board,
        player: Player,
        ticket_options: list[Ticket],
        is_initial_turn: bool,
    ) -> list[Ticket]:
        pass

    # Returning None means drawing from the deck.
    @abc.abstractmethod
    def draw_card(self, board: Board, player: Player, can_draw_any: bool) -> Card | None:
        pass

    @abc.abstractmethod
    def build_route(self, board: Board, player: Player) -> RouteInfo:
        pass


class UniformRandomPolicy(Policy):
    def choose_action(self, board: Board, player: Player) -> ActionType:
        return random.choice(get_valid_actions(board=board, player=player))

    def choose_tickets(
        self,
        board: Board,
        player: Player,
        ticket_options: list[Ticket],
        is_initial_turn: bool,
    ) -> list[Ticket]:
        draw_options = [
            ticket_options,
            *itertools.combinations(ticket_options, 2),
        ]
        if not is_initial_turn:
            draw_options.extend(itertools.combinations(draw_options, 1))

        return random.choice(draw_options)

    def draw_card(self, board: Board, player: Player, can_draw_any: bool) -> Card | None:
        card_options: set[Card | None] = {None}
        for card in board.visible_cards:
            if card.color != ANY or can_draw_any:
                card_options.add(card)

        return random.choice(list(card_options))

    def build_route(self, board: Board, player: Player) -> RouteInfo:
        route_options = get_build_route_options(board, player)
        return random.choice(route_options)


class Game:
    board: Board
    players: list[Player]
    policies: list[Policy]

    def __init__(self, policies: list[Policy]) -> None:
        assert 2 <= len(policies) <= 5
        self.policies = policies

        self.board = Board(num_players=len(policies))
        self.players = []

        for player_id, policy in enumerate(policies):
            player = Player(player_id=player_id)
            for _ in range(NUM_INITIAL_PLAYER_CARDS):
                card = self.board.card_deck.get()
                player.card_counts[card.color] += 1
                self.players.append(player)

            player.tickets = policy.choose_tickets(
                board=self.board,
                player=player,
                ticket_options=self.board.ticket_deck.get(),
                is_initial_turn=True
            )

    def run(self) -> None:
        final_player_id = None
        player_id = 0
        while True:
            policy = self.policies[player_id]
            player = self.players[player_id]

            action_type = policy.choose_action(self.board, player)
            if action_type == ActionType.DRAW_CARDS:
                pass
            elif action_type == ActionType.DRAW_TICKETS:
                tickets = self.board.ticket_deck.get()
                selected_tickets = policy.choose_tickets(
                    board=self.board,
                    player=player,
                    ticket_options=tickets,
                    is_initial_turn=False,
                )
                player.tickets.extend(selected_tickets)
            elif action_type == ActionType.BUILD_ROUTE:
                pass
            else:
                raise ValueError(f"Unknown action type: {action_type}")

            if final_player_id is None:
                if player_id == final_player_id:
                    break
            elif self.board.train_cars[player_id] <= NUM_LAST_TURN_CARS:
                final_player_id = player_id

            player_id = (player_id + 1) % len(self.players)

