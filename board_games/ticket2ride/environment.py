from dataclasses import dataclass

from termcolor import colored

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
from board_games.ticket2ride.color import ANY
from board_games.ticket2ride.consts import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    NUM_INITIAL_PLAYER_CARDS,
    NUM_LAST_TURN_CARS,
)
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.longest_path import find_longest_path
from board_games.ticket2ride.player import Player
from board_games.ticket2ride.policies import Policy
from board_games.ticket2ride.route import ROUTES, Route
from board_games.ticket2ride.state import (
    ObservedState,
    PlayerScore,
    Score,
    Transition
)
from board_games.ticket2ride.ticket import DrawnTickets


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

    def __init__(self, num_players: int) -> None:
        self.num_players = num_players
        self.reset()

    def reset(self) -> ObservedState:
        self.board = Board(num_players=self.num_players)
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
            if disjoint_sets.are_connected(ticket.source_city, ticket.destination_city):
                score.ticket_points += ticket.value
                score.completed_tickets += 1
            else:
                score.ticket_points -= ticket.value

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
        next_action_type: ActionType,
        drawn_tickets: DrawnTickets | None = None,
    ) -> Transition:
        # We must compute the score before advancing the state to the next player_id.
        score = self.get_score()
        return Transition(
            state=self.get_next_state(next_action_type, drawn_tickets), score=score)

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

            if (action.card is None or card.color != ANY) and self.consecutive_card_draws <= 1:
                return self.transition(next_action_type=ActionType.DRAW_CARD)
            else:
                return self.transition(next_action_type=ActionType.PLAN)
        elif action.action_type == ActionType.DRAW_TICKETS:
            assert isinstance(action, DrawTickets)
            player.tickets.extend(action.tickets)

            return self.transition(next_action_type=ActionType.PLAN)
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

            return self.transition(next_action_type=ActionType.PLAN)


class Roller:
    env: Environment
    policies: list[Policy]
    action_log: list[Action]

    def __init__(self, env: Environment, policies: list[Policy]):
        assert MIN_PLAYERS <= len(policies) <= MAX_PLAYERS
        self.env = env
        self.policies = policies
        self.action_log = []

    def run(self, verbose: bool = False) -> GameStats:
        initial_state = state = self.env.reset()
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
            self.action_log.append(action)

            transition = self.env.step(action)
            state = transition.state
            transitions.append(transition)

        assert transition is not None
        return GameStats(
            initial_state=initial_state,
            transitions=transitions,
        )
