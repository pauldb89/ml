import abc
import copy
import itertools
import random
from collections import defaultdict
from dataclasses import dataclass

from board_games.ticket2ride.consts import NUM_LAST_TURN_CARS, NUM_INITIAL_PLAYER_CARDS, ANY, \
    ROUTES, NUM_VISIBLE_CARDS, NUM_INITIAL_TRAIN_CARS, COLORS, NUM_COLOR_CARDS, NUM_ANY_CARDS, \
    TICKETS, MAX_VISIBLE_ANY_CARDS, LONGEST_PATH_POINTS
from board_games.ticket2ride.data_models import Card, Ticket, Color, ActionType, Action, DrawCards, \
    DrawTickets, BuildRoute, RouteInfo
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.longest_path import find_longest_path


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

    # TODO(pauldb): Keep track of visible points.
    def __init__(self, num_players: int) -> None:
        self.route_ownership = {}
        self.ticket_deck = TicketDeck()
        self.card_deck = CardDeck()

        self.train_cars = [NUM_INITIAL_TRAIN_CARS for _ in range(num_players)]
        self.route_points = [0 for _ in range(num_players)]

        self.visible_cards = []
        self.reveal_cards()

    def reveal_cards(self) -> None:
        while True:
            while len(self.visible_cards) < NUM_VISIBLE_CARDS:
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
        self.card_counts = card_counts or defaultdict(int)
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
    if len(board.card_deck) >= 2:
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
            draw_options.extend(itertools.combinations(ticket_options, 1))

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


@dataclass
class GameStats:
    route_points: list[int]
    ticket_points: list[int]
    longest_path_points: list[int]
    total_points: list[int]


class Game:
    board: Board
    players: list[Player]
    policies: list[Policy]
    action_log: list[Action]

    def __init__(self, policies: list[Policy]) -> None:
        assert 2 <= len(policies) <= 5
        self.policies = policies
        self.action_log = []

        self.board = Board(num_players=len(policies))
        self.players = []

        for player_id, policy in enumerate(policies):
            player = Player(player_id=player_id)
            for _ in range(NUM_INITIAL_PLAYER_CARDS):
                card = self.board.card_deck.draw()
                player.card_counts[card.color] += 1

            player.tickets = policy.choose_tickets(
                board=self.board,
                player=player,
                ticket_options=self.board.ticket_deck.get(),
                is_initial_turn=True
            )
            self.players.append(player)

    def run(self) -> GameStats:
        final_player_id = None
        player_id = 0
        while True:
            policy = self.policies[player_id]
            player = self.players[player_id]

            action_type = policy.choose_action(self.board, player)
            if action_type == ActionType.DRAW_CARDS:
                drawn_cards = []
                for can_draw_any in [True, False]:
                    card = policy.draw_card(self.board, player, can_draw_any=can_draw_any)
                    drawn_cards.append(card)

                    if card is None:
                        card = self.board.card_deck.draw()
                        player.card_counts[card.color] += 1
                    else:
                        self.board.visible_cards.remove(card)
                        player.card_counts[card.color] += 1
                        self.board.reveal_cards()

                        if card.color == ANY:
                            break

                self.action_log.append(
                    DrawCards(
                        player_id=player_id,
                        action_type=action_type,
                        cards=drawn_cards,
                    )
                )
            elif action_type == ActionType.DRAW_TICKETS:
                tickets = self.board.ticket_deck.get()
                selected_tickets = policy.choose_tickets(
                    board=self.board,
                    player=player,
                    ticket_options=tickets,
                    is_initial_turn=False,
                )
                player.tickets.extend(selected_tickets)
                self.action_log.append(
                    DrawTickets(
                        player_id=player_id,
                        action_type=action_type,
                        tickets=selected_tickets,
                    )
                )
            elif action_type == ActionType.BUILD_ROUTE:
                route_info = policy.build_route(self.board, player)

                assert route_info.route_id not in self.board.route_ownership
                self.board.route_ownership[route_info.route_id] = route_info

                route = ROUTES[route_info.route_id]
                num_regular_cards = route.length - route_info.num_any_cards
                for _ in range(num_regular_cards):
                    player.card_counts[route_info.color] -= 1
                    self.board.card_deck.discard(Card(color=route_info.color))

                for _ in range(route_info.num_any_cards):
                    player.card_counts[ANY] -= 1
                    self.board.card_deck.discard(Card(color=ANY))

                assert self.board.train_cars[player.id] >= route.length
                self.board.train_cars[player.id] -= route.length

                self.board.route_points[player.id] += route.value

                self.action_log.append(
                    BuildRoute(
                        player_id=player_id,
                        action_type=action_type,
                        route_info=route_info,
                    )
                )
            else:
                raise ValueError(f"Unknown action type: {action_type}")

            if final_player_id is not None:
                if player_id == final_player_id:
                    break
            elif self.board.train_cars[player_id] <= NUM_LAST_TURN_CARS:
                final_player_id = player_id

            player_id = (player_id + 1) % len(self.players)

        return self.compute_game_stats()

    def compute_game_stats(self) -> GameStats:
        longest_paths = []
        for player in self.players:
            routes = []
            for route_info in self.board.route_ownership.values():
                if route_info.player_id == player.id:
                    routes.append(ROUTES[route_info.route_id])

            longest_paths.append(find_longest_path(routes))

        max_path_length = max(longest_paths)
        route_points = []
        ticket_points = []
        longest_path_points = []
        for player in self.players:
            route_points.append(self.board.route_points[player.id])
            ticket_points.append(count_ticket_points(board=self.board, player=player))
            longest_path_points.append(
                LONGEST_PATH_POINTS if max_path_length == longest_paths[player.id] else 0
            )

        total_points = []
        for player_id in range(len(self.players)):
            total_points.append(
                route_points[player_id] + ticket_points[player_id] + longest_path_points[player_id]
            )

        return GameStats(
            route_points=route_points,
            ticket_points=ticket_points,
            longest_path_points=longest_path_points,
            total_points=total_points,
        )
