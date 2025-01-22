import abc
import enum
from dataclasses import dataclass
from typing import Iterable

from board_games.ticket2ride.board_logic import Player, Board
from board_games.ticket2ride.consts import ROUTES, EXTENDED_COLORS, CITIES, TICKETS
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.entities import Ticket, DrawnTickets


class TaskType(enum.IntEnum):
    CHOOSE_ACTION = 0
    DRAW_CARD = 1
    CHOOSE_TICKETS = 2
    BUILD_ROUTE = 3


class Separator(enum.IntEnum):
    # Secondary separators.
    ROUTE_INFO = 0
    TICKET = 1
    # Primary separators.
    TASK_TYPE = 2
    PLAYER_ID = 3
    TRAIN_CARS = 4
    VISIBLE_CARDS = 5
    OWNED_ROUTES = 6
    AVAILABLE_ROUTES = 7
    OWNED_TICKETS = 8
    DRAWN_TICKETS = 9


class FeatureType(enum.IntEnum):
    SEPARATOR = 0
    TASK_TYPE = 1
    PLAYER_ID = 2
    TRAIN_CARS = 3
    COLOR = 4
    CITY = 5
    ROUTE = 6
    TICKET = 7
    TICKET_POINTS = 8
    TICKET_STATUS = 9
    ROUTE_LENGTH = 10


@dataclass
class FeatureDef:
    type: FeatureType
    cardinality: int


FEATURE_REGISTRY: dict[FeatureType, FeatureDef] = {
    FeatureType.SEPARATOR: FeatureDef(type=FeatureType.SEPARATOR, cardinality=len(Separator)),
    FeatureType.TASK_TYPE: FeatureDef(type=FeatureType.TASK_TYPE, cardinality=len(TaskType)),
    FeatureType.PLAYER_ID: FeatureDef(type=FeatureType.PLAYER_ID, cardinality=2),
    FeatureType.TRAIN_CARS: FeatureDef(type=FeatureType.TRAIN_CARS, cardinality=46),
    FeatureType.COLOR: FeatureDef(type=FeatureType.COLOR, cardinality=len(EXTENDED_COLORS)),
    FeatureType.CITY: FeatureDef(type=FeatureType.CITY, cardinality=len(CITIES)),
    FeatureType.ROUTE: FeatureDef(type=FeatureType.ROUTE, cardinality=len(ROUTES)),
    FeatureType.TICKET: FeatureDef(type=FeatureType.TICKET, cardinality=len(TICKETS)),
    FeatureType.TICKET_POINTS: FeatureDef(type=FeatureType.TICKET_POINTS, cardinality=23),
    FeatureType.TICKET_STATUS: FeatureDef(type=FeatureType.TICKET_STATUS, cardinality=2),
    FeatureType.ROUTE_LENGTH: FeatureDef(type=FeatureType.ROUTE_LENGTH, cardinality=7),
}


@dataclass
class FeatureValue:
    type: FeatureType
    value: int


Features = list[FeatureValue]
BatchFeatures = list[Features]


@dataclass
class FeatureContext:
    board: Board
    player: Player
    task_type: TaskType
    drawn_tickets: DrawnTickets | None = None


class Extractor:
    @property
    @abc.abstractmethod
    def feature_types(self) -> list[FeatureType]:
        pass

    @abc.abstractmethod
    def extract(self, context: FeatureContext) -> Features:
        pass


class TaskTypeExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.TASK_TYPE]

    def extract(self, context: FeatureContext) -> Features:
        return [
            FeatureValue(type=FeatureType.SEPARATOR, value=Separator.TASK_TYPE),
            FeatureValue(type=FeatureType.TASK_TYPE, value=context.task_type),
        ]


class PlayerIdExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.PLAYER_ID]

    def extract(self, context: FeatureContext) -> Features:
        return [
            FeatureValue(type=FeatureType.SEPARATOR, value=Separator.PLAYER_ID),
            FeatureValue(type=FeatureType.PLAYER_ID, value=context.player.id),
        ]


class TrainCarsExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.TRAIN_CARS]

    def extract(self, context: FeatureContext) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=Separator.TRAIN_CARS)]
        for player_train_cars in context.board.train_cars:
            features.append(FeatureValue(type=FeatureType.TRAIN_CARS, value=player_train_cars))

        return features


class VisibleCardsExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.COLOR]

    def extract(self, context: FeatureContext) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=Separator.VISIBLE_CARDS)]
        for card in context.board.visible_cards:
            features.append(FeatureValue(type=FeatureType.COLOR, value=card.color.id))

        return features


class RouteOwnershipExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [
            FeatureType.SEPARATOR,
            FeatureType.ROUTE,
            FeatureType.CITY,
            FeatureType.PLAYER_ID,
        ]

    def extract(self, context: FeatureContext) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=Separator.OWNED_ROUTES)]
        for route_info in context.board.route_ownership.values():
            route = ROUTES[route_info.route_id]
            features.extend([
                FeatureValue(type=FeatureType.SEPARATOR, value=Separator.ROUTE_INFO),
                FeatureValue(type=FeatureType.ROUTE, value=route.id),
                FeatureValue(type=FeatureType.CITY, value=route.source_city.id),
                FeatureValue(type=FeatureType.CITY, value=route.destination_city.id),
                FeatureValue(type=FeatureType.PLAYER_ID, value=route_info.player_id),
            ])

        features.append(FeatureValue(type=FeatureType.SEPARATOR, value=Separator.AVAILABLE_ROUTES))
        for route in ROUTES:
            if route.id not in context.board.route_ownership:
                features.extend([
                    FeatureValue(type=FeatureType.SEPARATOR, value=Separator.ROUTE_INFO),
                    FeatureValue(type=FeatureType.ROUTE, value=route.id),
                    FeatureValue(type=FeatureType.CITY, value=route.source_city.id),
                    FeatureValue(type=FeatureType.CITY, value=route.destination_city.id),
                    FeatureValue(type=FeatureType.COLOR, value=route.color.id),
                    FeatureValue(type=FeatureType.ROUTE_LENGTH, value=route.length)
                ])

        return features


class TicketExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [
            FeatureType.SEPARATOR,
            FeatureType.TICKET,
            FeatureType.CITY,
            FeatureType.TICKET_POINTS,
            FeatureType.TICKET_STATUS,
        ]

    def extract(self, context: FeatureContext) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=self.separator)]

        disjoint_sets = DisjointSets()
        for route_info in context.board.route_ownership.values():
            if route_info.player_id == context.player.id:
                route = ROUTES[route_info.route_id]
                disjoint_sets.connect(route.source_city, route.destination_city)

        for ticket in self.get_tickets(context):
            features.extend([
                FeatureValue(type=FeatureType.SEPARATOR, value=Separator.TICKET),
                FeatureValue(type=FeatureType.TICKET, value=ticket.id),
                FeatureValue(type=FeatureType.CITY, value=ticket.source_city.id),
                FeatureValue(type=FeatureType.CITY, value=ticket.destination_city.id),
                FeatureValue(type=FeatureType.TICKET_POINTS, value=ticket.value),
                FeatureValue(
                    type=FeatureType.TICKET_STATUS,
                    value=disjoint_sets.are_connected(ticket.source_city, ticket.destination_city),
                )
            ])

        return features

    @property
    @abc.abstractmethod
    def separator(self) -> Separator:
        pass

    @abc.abstractmethod
    def get_tickets(self, context: FeatureContext) -> Iterable[Ticket]:
        pass


class OwnedTicketsExtractor(TicketExtractor):
    @property
    def separator(self) -> Separator:
        return Separator.OWNED_TICKETS

    def get_tickets(self, context: FeatureContext) -> Iterable[Ticket]:
        return context.player.tickets


class DrawnTicketsExtractor(TicketExtractor):
    @property
    def separator(self) -> Separator:
        return Separator.DRAWN_TICKETS

    def get_tickets(self, context: FeatureContext) -> Iterable[Ticket]:
        assert context.drawn_tickets is not None
        return context.drawn_tickets


ALL_EXTRACTORS: list[Extractor] = [
    TaskTypeExtractor(),
    PlayerIdExtractor(),
    TrainCarsExtractor(),
    VisibleCardsExtractor(),
    RouteOwnershipExtractor(),
    OwnedTicketsExtractor(),
    DrawnTicketsExtractor(),
]
