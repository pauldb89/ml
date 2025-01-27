import abc
import enum
from dataclasses import dataclass
from typing import Iterable

from board_games.ticket2ride.city import CITIES
from board_games.ticket2ride.color import EXTENDED_COLORS
from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.actions import ActionType
from board_games.ticket2ride.route import ROUTES
from board_games.ticket2ride.state import ObservedState
from board_games.ticket2ride.ticket import TICKETS, Ticket


class Separator(enum.IntEnum):
    # Secondary separators.
    ROUTE_INFO = 0
    TICKET = 1
    # Primary separators.
    ACTION_TYPE = 2
    PLAYER_ID = 3
    TRAIN_CARS = 4
    VISIBLE_CARDS = 5
    OWNED_ROUTES = 6
    AVAILABLE_ROUTES = 7
    OWNED_TICKETS = 8
    DRAWN_TICKETS = 9


class FeatureType(enum.IntEnum):
    SEPARATOR = 0
    ACTION_TYPE = 1
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
    FeatureType.ACTION_TYPE: FeatureDef(type=FeatureType.ACTION_TYPE, cardinality=len(ActionType)),
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


class Extractor:
    @property
    @abc.abstractmethod
    def feature_types(self) -> list[FeatureType]:
        pass

    @abc.abstractmethod
    def extract(self, state: ObservedState) -> Features:
        pass


class TaskTypeExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.ACTION_TYPE]

    def extract(self, state: ObservedState) -> Features:
        return [
            FeatureValue(type=FeatureType.SEPARATOR, value=Separator.ACTION_TYPE.value),
            FeatureValue(
                type=FeatureType.ACTION_TYPE, value=list(ActionType).index(state.next_action)
            ),
        ]


class PlayerIdExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.PLAYER_ID]

    def extract(self, state: ObservedState) -> Features:
        return [
            FeatureValue(type=FeatureType.SEPARATOR, value=Separator.PLAYER_ID.value),
            FeatureValue(type=FeatureType.PLAYER_ID, value=state.player.id),
        ]


class TrainCarsExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.TRAIN_CARS]

    def extract(self, state: ObservedState) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=Separator.TRAIN_CARS.value)]
        for player_train_cars in state.board.train_cars:
            features.append(FeatureValue(type=FeatureType.TRAIN_CARS, value=player_train_cars))

        return features


class VisibleCardsExtractor(Extractor):
    @property
    def feature_types(self) -> list[FeatureType]:
        return [FeatureType.SEPARATOR, FeatureType.COLOR]

    def extract(self, state: ObservedState) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=Separator.VISIBLE_CARDS.value)]
        for card in state.board.visible_cards:
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
            FeatureType.ROUTE_LENGTH,
        ]

    def extract(self, state: ObservedState) -> Features:
        features = [FeatureValue(type=FeatureType.SEPARATOR, value=Separator.OWNED_ROUTES.value)]
        for route_info in state.board.route_ownership.values():
            route = ROUTES[route_info.route_id]
            features.extend([
                FeatureValue(type=FeatureType.SEPARATOR, value=Separator.ROUTE_INFO.value),
                FeatureValue(type=FeatureType.ROUTE, value=route.id),
                FeatureValue(type=FeatureType.CITY, value=route.source_city.id),
                FeatureValue(type=FeatureType.CITY, value=route.destination_city.id),
                FeatureValue(type=FeatureType.PLAYER_ID, value=route_info.player_id),
            ])

        features.append(FeatureValue(type=FeatureType.SEPARATOR, value=Separator.AVAILABLE_ROUTES.value))
        for route in ROUTES:
            if route.id not in state.board.route_ownership:
                features.extend([
                    FeatureValue(type=FeatureType.SEPARATOR, value=Separator.ROUTE_INFO.value),
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

    def extract(self, state: ObservedState) -> Features:
        tickets = self.get_tickets(state)
        if tickets is None:
            return []

        disjoint_sets = DisjointSets()
        for route_info in state.board.route_ownership.values():
            if route_info.player_id == state.player.id:
                route = ROUTES[route_info.route_id]
                disjoint_sets.connect(route.source_city, route.destination_city)

        features = [FeatureValue(type=FeatureType.SEPARATOR, value=self.separator.value)]
        for ticket in tickets:
            features.extend([
                FeatureValue(type=FeatureType.SEPARATOR, value=Separator.TICKET.value),
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
    def get_tickets(self, state: ObservedState) -> Iterable[Ticket] | None:
        pass


class OwnedTicketsExtractor(TicketExtractor):
    @property
    def separator(self) -> Separator:
        return Separator.OWNED_TICKETS

    def get_tickets(self, state: ObservedState) -> list[Ticket] | None:
        return state.player.tickets


class DrawnTicketsExtractor(TicketExtractor):
    @property
    def separator(self) -> Separator:
        return Separator.DRAWN_TICKETS

    def get_tickets(self, state: ObservedState) -> Iterable[Ticket] | None:
        return state.drawn_tickets


ALL_EXTRACTORS: list[Extractor] = [
    TaskTypeExtractor(),
    PlayerIdExtractor(),
    TrainCarsExtractor(),
    VisibleCardsExtractor(),
    RouteOwnershipExtractor(),
    OwnedTicketsExtractor(),
    DrawnTicketsExtractor(),
]
