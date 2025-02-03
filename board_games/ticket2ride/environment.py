import collections
import random
from dataclasses import dataclass

from termcolor import colored

from board_games.ticket2ride.action_utils import get_draw_card_options
from board_games.ticket2ride.actions import (
    Action,
    ActionType,
    BuildRoute,
    DrawCard,
    DrawTickets,
    Plan
)
from board_games.ticket2ride.board import Board
from board_games.ticket2ride.card import Card
from board_games.ticket2ride.color import ANY, Color, COLORS
from board_games.ticket2ride.consts import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    NUM_INITIAL_PLAYER_CARDS,
    NUM_LAST_TURN_CARS, NUM_COLOR_CARDS, NUM_ANY_CARDS,
)
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.longest_path import find_longest_path
from board_games.ticket2ride.player import Player
from board_games.ticket2ride.policies import Policy
from board_games.ticket2ride.render_utils import print_player
from board_games.ticket2ride.route import ROUTES, Route
from board_games.ticket2ride.state import (
    ObservedState,
    PlayerScore,
    Score,
    Transition
)
from board_games.ticket2ride.ticket import DrawnTickets


def verify_card_bookkeeping(board: Board, players: list[Player]) -> None:
    card_counts: dict[Color, int] = collections.defaultdict(int)
    for player in players:
        for color, cnt in player.card_counts.items():
            card_counts[color] += cnt

    for card in board.visible_cards:
        card_counts[card.color] += 1

    for card in board.card_deck.deck + board.card_deck.discard_pile:
        card_counts[card.color] += 1

    for color in COLORS:
        assert card_counts[color] == NUM_COLOR_CARDS
    assert card_counts[ANY] == NUM_ANY_CARDS


@dataclass
class GameStats:
    initial_state: ObservedState
    transitions: list[Transition]

    @property
    def scorecard(self) -> list[PlayerScore]:
        return self.transitions[-1].score.scorecard


class Environment:
    board: Board
    players: list[Player]

    turn_id: int
    player_id: int
    action_type: ActionType
    game_over: bool

    final_player_id: int | None
    consecutive_card_draws: int
    longest_path_length: int

    scorecard: list[PlayerScore]
    rng: random.Random

    def __init__(self, num_players: int, seed: int = 0) -> None:
        self.num_players = num_players
        self.reset(seed)

    def reset(self, seed: int) -> ObservedState:
        self.board = Board(num_players=self.num_players, rng=random.Random(seed))
        self.players = []
        self.scorecard = []

        for player_id in range(self.num_players):
            player = Player(player_id=player_id)
            for _ in range(NUM_INITIAL_PLAYER_CARDS):
                card = self.board.card_deck.draw()
                player.card_counts[card.color] += 1
            self.players.append(player)
            self.scorecard.append(PlayerScore(player_id=player_id))

        self.turn_id = 0
        self.player_id = 0
        self.action_type = ActionType.PLAN
        self.game_over = False

        self.final_player_id = None
        self.consecutive_card_draws = 0
        self.longest_path_length = 0

        verify_card_bookkeeping(self.board, self.players)

        return ObservedState(
            board=self.board,
            player=self.players[self.player_id],
            action_type=ActionType.PLAN,
            # TODO(pauldb): Remember to update the valid action space transition to only include drawing tickets in the first turn of the game.
            turn_id=self.turn_id,
            terminal=self.game_over,
            consecutive_card_draws=self.consecutive_card_draws,
        )

    def get_score(self) -> Score:
        if self.action_type not in (ActionType.DRAW_TICKETS, ActionType.BUILD_ROUTE):
            return Score(scorecard=self.scorecard, turn_score=PlayerScore(self.player_id))

        score = PlayerScore(player_id=self.player_id)
        disjoint_sets = DisjointSets()
        owned_routes: list[Route] = []
        for route_info in self.board.route_ownership.values():
            if route_info.player_id == self.player_id:
                route = ROUTES[route_info.route_id]
                disjoint_sets.connect(route.source_city, route.destination_city)

                owned_routes.append(route)
                score.route_points += route.value

        for ticket in self.players[self.player_id].tickets:
            score.total_tickets += 1
            if disjoint_sets.are_connected(ticket.source_city, ticket.destination_city):
                score.ticket_points += ticket.value
                score.completed_tickets += 1
            else:
                score.ticket_points -= ticket.value

        for route in owned_routes:
            score.owned_routes_by_length[route.length] += 1

        score.longest_path = find_longest_path(owned_routes)
        if score.longest_path > self.longest_path_length:
            self.longest_path_length = score.longest_path

        score.longest_path_bonus = score.longest_path == self.longest_path_length

        prev_score = self.scorecard[self.player_id]
        self.scorecard[self.player_id] = score
        return Score(
            scorecard=self.scorecard,
            turn_score=PlayerScore(
                player_id=self.player_id,
                route_points=score.route_points - prev_score.route_points,
                ticket_points=score.ticket_points - prev_score.ticket_points,
                completed_tickets=score.completed_tickets - prev_score.completed_tickets,
                longest_path_bonus=score.longest_path_bonus and not prev_score.longest_path_bonus,
                longest_path=score.longest_path - prev_score.longest_path,
            )
        )

    def get_next_state(
        self,
        next_action_type: ActionType,
        drawn_tickets: DrawnTickets | None = None,
    ) -> ObservedState:
        if next_action_type != ActionType.DRAW_CARD:
            self.consecutive_card_draws = 0

        if next_action_type == ActionType.PLAN:
            if self.final_player_id is not None:
                self.game_over = self.player_id == self.final_player_id
            elif self.board.train_cars[self.player_id] <= NUM_LAST_TURN_CARS:
                self.final_player_id = self.player_id

            if self.player_id + 1 < len(self.players):
                self.player_id += 1
            else:
                self.player_id = 0
                self.turn_id += 1

        self.action_type = next_action_type

        return ObservedState(
            board=self.board,
            player=self.players[self.player_id],
            action_type=next_action_type,
            drawn_tickets=drawn_tickets,
            # TODO(pauldb): Remember to update the valid action space transition to only include drawing tickets in the first turn of the game.
            turn_id=self.turn_id,
            terminal=self.game_over,
            consecutive_card_draws=self.consecutive_card_draws,
        )

    def transition(
        self,
        action: Action,
        next_action_type: ActionType,
        drawn_tickets: DrawnTickets | None = None,
    ) -> Transition:
        verify_card_bookkeeping(self.board, self.players)

        return Transition(
            # We must compute the score before advancing the state to the next player_id.
            score=self.get_score(),
            state=self.get_next_state(next_action_type, drawn_tickets),
            action=action,
        )

    def step(self, action: Action) -> Transition:
        assert action.action_type == self.action_type
        assert action.player_id == self.player_id
        assert not self.game_over

        player = self.players[self.player_id]

        if action.action_type == ActionType.PLAN:
            assert isinstance(action, Plan)
            drawn_tickets = None
            if action.next_action_type == ActionType.DRAW_TICKETS:
                drawn_tickets = self.board.ticket_deck.get()

            return self.transition(
                action=action,
                next_action_type=action.next_action_type,
                drawn_tickets=drawn_tickets
            )
        elif action.action_type == ActionType.DRAW_CARD:
            assert isinstance(action, DrawCard)
            card = action.card
            if card is None:
                card = self.board.card_deck.draw()
            else:
                self.board.visible_cards.remove(card)
                self.board.reveal_cards()

            player.card_counts[card.color] += 1
            self.consecutive_card_draws += 1

            if (
                self.consecutive_card_draws <= 1
                and (action.card is None or card.color != ANY)
                and get_draw_card_options(self.board, self.consecutive_card_draws)
            ):
                return self.transition(action=action, next_action_type=ActionType.DRAW_CARD)
            else:
                return self.transition(action=action, next_action_type=ActionType.PLAN)
        elif action.action_type == ActionType.DRAW_TICKETS:
            assert isinstance(action, DrawTickets)
            player.tickets.extend(action.tickets)

            return self.transition(action=action, next_action_type=ActionType.PLAN)
        elif action.action_type == ActionType.BUILD_ROUTE:
            assert isinstance(action, BuildRoute)

            route_info = action.route_info
            assert route_info.route_id not in self.board.route_ownership
            self.board.route_ownership[route_info.route_id] = route_info

            route = ROUTES[route_info.route_id]
            num_regular_cards = route.length - route_info.num_any_cards
            for _ in range(num_regular_cards):
                player.card_counts[route_info.color] -= 1
                assert player.card_counts[route_info.color] >= 0
                self.board.card_deck.discard(Card(color=route_info.color))

            for _ in range(route_info.num_any_cards):
                player.card_counts[ANY] -= 1
                assert player.card_counts[ANY] >= 0
                self.board.card_deck.discard(Card(color=ANY))

            assert self.board.train_cars[player.id] >= route.length
            self.board.train_cars[player.id] -= route.length

            self.board.route_points[player.id] += route.value

            return self.transition(action=action, next_action_type=ActionType.PLAN)


class Roller:
    env: Environment
    policies: list[Policy]

    def __init__(self, env: Environment, policies: list[Policy]):
        assert MIN_PLAYERS <= len(policies) <= MAX_PLAYERS
        self.env = env
        self.policies = policies

    def run(self, seed: int, verbose: bool = False) -> GameStats:
        initial_state = state = self.env.reset(seed)
        transition = None
        transitions = []
        while not state.terminal:
            policy = self.policies[state.player.id]
            action = policy.choose_action(state)
            if verbose:
                print(
                    colored(f"Action turn {state.turn_id}: ", color="green", attrs=["bold"])
                    + str(action)
                )

            transition = self.env.step(action)
            state = transition.state
            transitions.append(transition)

        assert transition is not None
        return GameStats(
            initial_state=initial_state,
            transitions=transitions,
        )
