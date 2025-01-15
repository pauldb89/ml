from board_games.ticket2ride.board_logic import Board, Player
from board_games.ticket2ride.consts import ROUTES, ANY, COLORS
from board_games.ticket2ride.data_models import RouteInfo, ActionType


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


def get_valid_actions(board: Board, player: Player) -> list[ActionType]:
    valid_action_types = []
    if len(board.card_deck) + len(board.visible_cards) >= 2:
        valid_action_types.append(ActionType.DRAW_CARDS)

    if len(board.ticket_deck) >= 3:
        valid_action_types.append(ActionType.DRAW_TICKETS)

    build_route_options = get_build_route_options(board, player)
    if len(build_route_options) > 0:
        valid_action_types.append(ActionType.BUILD_ROUTE)

    return valid_action_types


