from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class City:
    id: int
    name: str


VANCOUVER = City(id=0, name="Vancouver")
CALGARY = City(id=1, name="Calgary")
WINNIPEG = City(id=2, name="Winnipeg")
SAULT_ST_MARIE = City(id=3, name="Sault St. Marie")
MONTREAL = City(id=4, name="Montreal")
SEATTLE = City(id=5, name="Seattle")
TORONTO = City(id=6, name="Toronto")
BOSTON = City(id=7, name="Boston")
PORTLAND = City(id=8, name="Portland")
HELENA = City(id=9, name="Helena")
DULUTH = City(id=10, name="Duluth")
NEW_YORK = City(id=11, name="New York")
CHICAGO = City(id=12, name="Chicago")
PITTSBURGH = City(id=13, name="Pittsburgh")
OMAHA = City(id=14, name="Omaha")
WASHINGTON = City(id=15, name="Washington")
SALT_LAKE_CITY = City(id=16, name="Salt Lake City")
DENVER = City(id=17, name="Denver")
KANSAS_CITY = City(id=18, name="Kansas City")
SAINT_LOUIS = City(id=19, name="Saint Louis")
NASHVILLE = City(id=20, name="Nashville")
RALEIGH = City(id=21, name="Raleigh")
SAN_FRANCISCO = City(id=22, name="San Francisco")
LAS_VEGAS = City(id=23, name="Las Vegas")
SANTA_FE = City(id=24, name="Santa Fe")
OKLAHOMA_CITY = City(id=25, name="Oklahoma City")
LITTLE_ROCK = City(id=26, name="Little Rock")
ATLANTA = City(id=27, name="Atlanta")
CHARLESTON = City(id=28, name="Charleston")
LOS_ANGELES = City(id=29, name="Los Angeles")
PHOENIX = City(id=30, name="Phoenix")
EL_PASO = City(id=31, name="El Paso")
DALLAS = City(id=32, name="Dallas")
HOUSTON = City(id=33, name="Houston")
NEW_ORLEANS = City(id=34, name="New Orleans")
MIAMI = City(id=35, name="Miami")

CITIES: list[City] = [
    VANCOUVER,
    CALGARY,
    WINNIPEG,
    SAULT_ST_MARIE,
    MONTREAL,
    SEATTLE,
    TORONTO,
    BOSTON,
    PORTLAND,
    HELENA,
    DULUTH,
    NEW_YORK,
    CHICAGO,
    PITTSBURGH,
    OMAHA,
    WASHINGTON,
    SALT_LAKE_CITY,
    DENVER,
    KANSAS_CITY,
    SAINT_LOUIS,
    NASHVILLE,
    RALEIGH,
    SAN_FRANCISCO,
    LAS_VEGAS,
    SANTA_FE,
    OKLAHOMA_CITY,
    LITTLE_ROCK,
    ATLANTA,
    CHARLESTON,
    LOS_ANGELES,
    PHOENIX,
    EL_PASO,
    DALLAS,
    HOUSTON,
    NEW_ORLEANS,
    MIAMI
]
