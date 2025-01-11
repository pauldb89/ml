from board_games.data_models import City


def test_city() -> None:
    city = City(id=0, name="London")
    assert city.id == 0
    assert city.name == "London"
