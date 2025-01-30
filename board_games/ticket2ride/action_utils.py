import itertools

from board_games.ticket2ride.actions import ActionType
from board_games.ticket2ride.board import Board
from board_games.ticket2ride.card import Card
from board_games.ticket2ride.color import ANY, COLORS, EXTENDED_COLORS, Color
from board_games.ticket2ride.route import ROUTES, Route
from board_games.ticket2ride.route_info import RouteInfo
from board_games.ticket2ride.state import ObservedState
from board_games.ticket2ride.ticket import DrawnTickets, Tickets


def get_valid_actions(state: ObservedState) -> list[ActionType]:
    valid_action_types = []

    if len(state.board.ticket_deck) >= 3:
        valid_action_types.append(ActionType.DRAW_TICKETS)

    if state.turn_id > 0:
        if len(state.board.card_deck) + len(state.board.visible_cards) >= 2:
            valid_action_types.append(ActionType.DRAW_CARD)

        build_route_options = get_build_route_options(state)
        if len(build_route_options) > 0:
            valid_action_types.append(ActionType.BUILD_ROUTE)

    # If the player has no valid actions, they are forced to skip a turn. Setting action_type
    # to PLAN will pass the turn to the next player.
    if not valid_action_types:
        valid_action_types.append(ActionType.PLAN)

    return valid_action_types


def get_draw_card_options(board: Board, consecutive_card_draws: int) -> list[Card | None]:
    card_options: list[Card | None] = []
    if len(board.card_deck) >= 1:
        card_options.append(None)

    for card in board.visible_cards:
        if card in card_options:
            continue

        if card.color == ANY and consecutive_card_draws > 0:
            continue

        card_options.append(card)

    return card_options


def get_build_route_options(state: ObservedState) -> list[RouteInfo]:
    board, player = state.board, state.player

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


def get_ticket_draw_options(tickets: DrawnTickets, is_initial_turn: bool) -> list[Tickets]:
    start = 1 if is_initial_turn else 0
    
    draw_options: list[Tickets] = []
    for k in range(start, len(tickets)):
        draw_options.extend(itertools.combinations(tickets, k+1))
    return draw_options


def generate_card_classes() -> list[Card | None]:
    return [None] + [Card(c) for c in EXTENDED_COLORS]


def generate_choose_ticket_classes() -> list[tuple[int, ...]]:
    return [x for k in range(3) for x in itertools.combinations(range(3), k+1)]


def generate_build_route_classes() -> list[tuple[Route, Color]]:
    classes = []
    for route in ROUTES:
        colors = [route.color] if route.color != ANY else COLORS
        for color in colors:
            classes.append((route, color))
    return classes


PLAN_CLASSES: list[ActionType] = [
    ActionType.DRAW_CARD,
    ActionType.DRAW_TICKETS,
    ActionType.BUILD_ROUTE,
    ActionType.PLAN,
]
DRAW_CARD_CLASSES = generate_card_classes()
CHOOSE_TICKETS_CLASSES = generate_choose_ticket_classes()
BUILD_ROUTE_CLASSES = generate_build_route_classes()

