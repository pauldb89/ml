from board_games.ticket2ride.city import City, CITIES


class DisjointSets:
    def __init__(self):
        self.parent = list(range(len(CITIES)))

    def find_root(self, city_id: int) -> int:
        if self.parent[city_id] == city_id:
            return city_id

        self.parent[city_id] = self.find_root(self.parent[city_id])
        return self.parent[city_id]

    def connect(self, city1: City, city2: City) -> None:
        self.parent[self.find_root(city1.id)] = self.find_root(city2.id)

    def are_connected(self, city1: City, city2: City) -> bool:
        return self.find_root(city1.id) == self.find_root(city2.id)
