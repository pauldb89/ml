from dataclasses import dataclass

from board_games.ticket2ride.board_logic import Player, Board
from board_games.ticket2ride.consts import LONGEST_PATH_POINTS, ROUTES, NUM_LAST_TURN_CARS, ANY, \
    NUM_INITIAL_PLAYER_CARDS
from board_games.ticket2ride.data_models import Card, DrawTickets, ActionType, DrawCards, Action, \
    BuildRoute
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.longest_path import find_longest_path
from board_games.ticket2ride.policies import Policy


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
