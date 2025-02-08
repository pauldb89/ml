from board_games.ticket2ride.disjoint_sets import DisjointSets
from board_games.ticket2ride.city import SEATTLE, HELENA, LOS_ANGELES, PHOENIX, DENVER


def test_disjoint_sets() -> None:
    disjoint_sets = DisjointSets()

    assert not disjoint_sets.are_connected(SEATTLE, HELENA)

    disjoint_sets.connect(SEATTLE, HELENA)
    assert disjoint_sets.are_connected(SEATTLE, HELENA)

    assert not disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
    disjoint_sets.connect(LOS_ANGELES, PHOENIX)
    assert not disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
    disjoint_sets.connect(HELENA, DENVER)
    assert not disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
    disjoint_sets.connect(DENVER, PHOENIX)
    assert disjoint_sets.are_connected(LOS_ANGELES, SEATTLE)
