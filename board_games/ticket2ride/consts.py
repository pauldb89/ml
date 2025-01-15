from board_games.ticket2ride.data_models import City, Color, Route, Ticket

NUM_COLOR_CARDS = 12
NUM_ANY_CARDS = 14
NUM_VISIBLE_CARDS = 5
NUM_INITIAL_PLAYER_CARDS = 4
NUM_INITIAL_TRAIN_CARS = 45
NUM_LAST_TURN_CARS = 2
MAX_VISIBLE_ANY_CARDS = 2
LONGEST_PATH_POINTS = 10

PURPLE = Color(id=0, name="purple")
WHITE = Color(id=1, name="white")
BLUE = Color(id=2, name="blue")
YELLOW = Color(id=3, name="yellow")
ORANGE = Color(id=4, name="orange")
BLACK = Color(id=5, name="black")
RED = Color(id=6, name="red")
GREEN = Color(id=7, name="green")
ANY = Color(id=8, name="any")

COLORS = [
    PURPLE,
    WHITE,
    BLUE,
    YELLOW,
    ORANGE,
    BLACK,
    RED,
    GREEN,
]

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
PITTSBURGH = City(id=13, name="PittsBurgh")
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
    VANCOUVER, CALGARY, WINNIPEG, SAULT_ST_MARIE, MONTREAL, SEATTLE, TORONTO, BOSTON, PORTLAND,
    HELENA, DULUTH, NEW_YORK, CHICAGO, PITTSBURGH, OMAHA, WASHINGTON, SALT_LAKE_CITY, DENVER,
    KANSAS_CITY, SAINT_LOUIS, NASHVILLE, RALEIGH, SAN_FRANCISCO, LAS_VEGAS, SANTA_FE, OKLAHOMA_CITY,
    LITTLE_ROCK, ATLANTA, CHARLESTON, LOS_ANGELES, PHOENIX, EL_PASO, DALLAS, HOUSTON, NEW_ORLEANS,
    MIAMI
]

ROUTES: list[Route] = [
    Route(id=0, source_city=VANCOUVER, destination_city=CALGARY, color=ANY, length=3),
    Route(id=1, source_city=VANCOUVER, destination_city=SEATTLE, color=ANY, length=1),
    Route(id=2, source_city=VANCOUVER, destination_city=SEATTLE, color=ANY, length=1),
    Route(id=3, source_city=CALGARY, destination_city=WINNIPEG, color=WHITE, length=6),
    Route(id=4, source_city=CALGARY, destination_city=HELENA, color=ANY, length=4),
    Route(id=5, source_city=WINNIPEG, destination_city=SAULT_ST_MARIE, color=ANY, length=6),
    Route(id=6, source_city=WINNIPEG, destination_city=DULUTH, color=BLACK, length=4),
    Route(id=7, source_city=WINNIPEG, destination_city=HELENA, color=BLUE, length=4),
    Route(id=8, source_city=SAULT_ST_MARIE, destination_city=MONTREAL, color=BLACK, length=5),
    Route(id=9, source_city=SAULT_ST_MARIE, destination_city=TORONTO, color=ANY, length=2),
    Route(id=10, source_city=SAULT_ST_MARIE, destination_city=DULUTH, color=ANY, length=3),
    Route(id=11, source_city=MONTREAL, destination_city=TORONTO, color=ANY, length=3),
    Route(id=12, source_city=MONTREAL, destination_city=BOSTON, color=ANY, length=2),
    Route(id=13, source_city=MONTREAL, destination_city=BOSTON, color=ANY, length=2),
    Route(id=14, source_city=MONTREAL, destination_city=NEW_YORK, color=BLUE, length=3),
    Route(id=15, source_city=SEATTLE, destination_city=PORTLAND, color=ANY, length=1),
    Route(id=16, source_city=SEATTLE, destination_city=PORTLAND, color=ANY, length=1),
    Route(id=17, source_city=SEATTLE, destination_city=HELENA, color=YELLOW, length=6),
    Route(id=18, source_city=TORONTO, destination_city=PITTSBURGH, color=ANY, length=2),
    Route(id=19, source_city=TORONTO, destination_city=DULUTH, color=PURPLE, length=6),
    Route(id=20, source_city=TORONTO, destination_city=CHICAGO, color=WHITE, length=4),
    Route(id=21, source_city=BOSTON, destination_city=NEW_YORK, color=YELLOW, length=2),
    Route(id=22, source_city=BOSTON, destination_city=NEW_YORK, color=RED, length=2),
    Route(id=23, source_city=PORTLAND, destination_city=SAN_FRANCISCO, color=GREEN, length=5),
    Route(id=24, source_city=PORTLAND, destination_city=SAN_FRANCISCO, color=PURPLE, length=5),
    Route(id=25, source_city=SAN_FRANCISCO, destination_city=SALT_LAKE_CITY, color=ORANGE, length=5),
    Route(id=26, source_city=SAN_FRANCISCO, destination_city=SALT_LAKE_CITY, color=WHITE, length=5),
    Route(id=27, source_city=PORTLAND, destination_city=SALT_LAKE_CITY, color=BLUE, length=6),
    Route(id=28, source_city=HELENA, destination_city=DULUTH, color=ORANGE, length=6),
    Route(id=29, source_city=HELENA, destination_city=OMAHA, color=RED, length=5),
    Route(id=30, source_city=HELENA, destination_city=DENVER, color=GREEN, length=4),
    Route(id=31, source_city=HELENA, destination_city=SALT_LAKE_CITY, color=PURPLE, length=3),
    Route(id=32, source_city=DULUTH, destination_city=OMAHA, color=ANY, length=2),
    Route(id=33, source_city=DULUTH, destination_city=OMAHA, color=ANY, length=2),
    Route(id=34, source_city=DULUTH, destination_city=CHICAGO, color=RED, length=3),
    Route(id=35, source_city=NEW_YORK, destination_city=PITTSBURGH, color=WHITE, length=2),
    Route(id=36, source_city=NEW_YORK, destination_city=PITTSBURGH, color=GREEN, length=2),
    Route(id=37, source_city=NEW_YORK, destination_city=WASHINGTON, color=ORANGE, length=2),
    Route(id=38, source_city=NEW_YORK, destination_city=WASHINGTON, color=BLACK, length=2),
    Route(id=39, source_city=CHICAGO, destination_city=PITTSBURGH, color=ORANGE, length=3),
    Route(id=40, source_city=CHICAGO, destination_city=PITTSBURGH, color=BLACK, length=3),
    Route(id=41, source_city=CHICAGO, destination_city=SAINT_LOUIS, color=GREEN, length=2),
    Route(id=42, source_city=CHICAGO, destination_city=SAINT_LOUIS, color=WHITE, length=2),
    Route(id=43, source_city=CHICAGO, destination_city=OMAHA, color=BLUE, length=4),
    Route(id=44, source_city=PITTSBURGH, destination_city=SAINT_LOUIS, color=GREEN, length=5),
    Route(id=45, source_city=PITTSBURGH, destination_city=NASHVILLE, color=YELLOW, length=4),
    Route(id=46, source_city=PITTSBURGH, destination_city=RALEIGH, color=ANY, length=2),
    Route(id=47, source_city=OMAHA, destination_city=KANSAS_CITY, color=ANY, length=1),
    Route(id=48, source_city=OMAHA, destination_city=KANSAS_CITY, color=ANY, length=1),
    Route(id=49, source_city=OMAHA, destination_city=DENVER, color=PURPLE, length=4),
    Route(id=50, source_city=WASHINGTON, destination_city=PITTSBURGH, color=ANY, length=2),
    Route(id=51, source_city=WASHINGTON, destination_city=RALEIGH, color=ANY, length=2),
    Route(id=52, source_city=DENVER, destination_city=KANSAS_CITY, color=BLACK, length=4),
    Route(id=53, source_city=DENVER, destination_city=KANSAS_CITY, color=ORANGE, length=4),
    Route(id=54, source_city=DENVER, destination_city=OKLAHOMA_CITY, color=RED, length=4),
    Route(id=55, source_city=DENVER, destination_city=PHOENIX, color=WHITE, length=5),
    Route(id=56, source_city=DENVER, destination_city=SANTA_FE, color=ANY, length=2),
    Route(id=57, source_city=SANTA_FE, destination_city=OKLAHOMA_CITY, color=BLUE, length=3),
    Route(id=58, source_city=SANTA_FE, destination_city=PHOENIX, color=ANY, length=3),
    Route(id=59, source_city=KANSAS_CITY, destination_city=OKLAHOMA_CITY, color=ANY, length=2),
    Route(id=60, source_city=KANSAS_CITY, destination_city=OKLAHOMA_CITY, color=ANY, length=2),
    Route(id=61, source_city=KANSAS_CITY, destination_city=SAINT_LOUIS, color=BLUE, length=2),
    Route(id=62, source_city=KANSAS_CITY, destination_city=SAINT_LOUIS, color=PURPLE, length=2),
    Route(id=63, source_city=SAINT_LOUIS, destination_city=LITTLE_ROCK, color=ANY, length=2),
    Route(id=64, source_city=SAINT_LOUIS, destination_city=NASHVILLE, color=ANY, length=2),
    Route(id=65, source_city=NASHVILLE, destination_city=RALEIGH, color=BLACK, length=3),
    Route(id=66, source_city=NASHVILLE, destination_city=ATLANTA, color=ANY, length=1),
    Route(id=67, source_city=RALEIGH, destination_city=ATLANTA, color=ANY, length=2),
    Route(id=68, source_city=RALEIGH, destination_city=CHARLESTON, color=ANY, length=2),
    Route(id=69, source_city=SAN_FRANCISCO, destination_city=LOS_ANGELES, color=YELLOW, length=3),
    Route(id=70, source_city=SAN_FRANCISCO, destination_city=LOS_ANGELES, color=PURPLE, length=3),
    Route(id=71, source_city=LAS_VEGAS, destination_city=LOS_ANGELES, color=ANY, length=2),
    Route(id=72, source_city=SANTA_FE, destination_city=EL_PASO, color=ANY, length=2),
    Route(id=73, source_city=OKLAHOMA_CITY, destination_city=EL_PASO, color=YELLOW, length=5),
    Route(id=74, source_city=OKLAHOMA_CITY, destination_city=DALLAS, color=ANY, length=2),
    Route(id=75, source_city=OKLAHOMA_CITY, destination_city=DALLAS, color=ANY, length=2),
    Route(id=76, source_city=OKLAHOMA_CITY, destination_city=LITTLE_ROCK, color=ANY, length=2),
    Route(id=77, source_city=LITTLE_ROCK, destination_city=DALLAS, color=ANY, length=2),
    Route(id=78, source_city=LITTLE_ROCK, destination_city=NASHVILLE, color=WHITE, length=3),
    Route(id=79, source_city=LITTLE_ROCK, destination_city=NEW_ORLEANS, color=GREEN, length=3),
    Route(id=80, source_city=ATLANTA, destination_city=CHARLESTON, color=ANY, length=2),
    Route(id=81, source_city=ATLANTA, destination_city=NEW_ORLEANS, color=YELLOW, length=4),
    Route(id=82, source_city=ATLANTA, destination_city=NEW_ORLEANS, color=ORANGE, length=4),
    Route(id=83, source_city=ATLANTA, destination_city=MIAMI, color=BLUE, length=5),
    Route(id=84, source_city=CHARLESTON, destination_city=MIAMI, color=PURPLE, length=4),
    Route(id=85, source_city=LOS_ANGELES, destination_city=PHOENIX, color=ANY, length=3),
    Route(id=86, source_city=LOS_ANGELES, destination_city=EL_PASO, color=BLACK, length=6),
    Route(id=87, source_city=PHOENIX, destination_city=EL_PASO, color=ANY, length=3),
    Route(id=88, source_city=EL_PASO, destination_city=DALLAS, color=RED, length=4),
    Route(id=89, source_city=EL_PASO, destination_city=HOUSTON, color=GREEN, length=6),
    Route(id=90, source_city=DALLAS, destination_city=HOUSTON, color=ANY, length=1),
    Route(id=91, source_city=DALLAS, destination_city=HOUSTON, color=ANY, length=1),
    Route(id=92, source_city=HOUSTON, destination_city=NEW_ORLEANS, color=ANY, length=2),
    Route(id=93, source_city=NEW_ORLEANS, destination_city=MIAMI, color=RED, length=6),
    Route(id=94, source_city=CALGARY, destination_city=SEATTLE, color=ANY, length=4),
    Route(id=95, source_city=WASHINGTON, destination_city=RALEIGH, color=ANY, length=2),
    Route(id=96, source_city=SALT_LAKE_CITY, destination_city=DENVER, color=RED, length=3),
    Route(id=97, source_city=SALT_LAKE_CITY, destination_city=DENVER, color=YELLOW, length=3),
    Route(id=98, source_city=SALT_LAKE_CITY, destination_city=LAS_VEGAS, color=ORANGE, length=3),
    Route(id=99, source_city=RALEIGH, destination_city=ATLANTA, color=ANY, length=2),
]

TICKETS: list[Ticket] = [
    Ticket(id=0, source_city=DENVER, destination_city=EL_PASO, value=4),
    Ticket(id=1, source_city=KANSAS_CITY, destination_city=HOUSTON, value=5),
    Ticket(id=2, source_city=NEW_YORK, destination_city=ATLANTA, value=6),
    Ticket(id=3, source_city=CALGARY, destination_city=SALT_LAKE_CITY, value=7),
    Ticket(id=4, source_city=CHICAGO, destination_city=NEW_ORLEANS, value=7),
    Ticket(id=5, source_city=HELENA, destination_city=LOS_ANGELES, value=8),
    Ticket(id=6, source_city=DULUTH, destination_city=HOUSTON, value=8),
    Ticket(id=7, source_city=SAULT_ST_MARIE, destination_city=NASHVILLE, value=8),
    Ticket(id=8, source_city=SEATTLE, destination_city=LOS_ANGELES, value=9),
    Ticket(id=9, source_city=MONTREAL, destination_city=ATLANTA, value=9),
    Ticket(id=10, source_city=CHICAGO, destination_city=SANTA_FE, value=9),
    Ticket(id=11, source_city=SAULT_ST_MARIE, destination_city=OKLAHOMA_CITY, value=9),
    Ticket(id=12, source_city=TORONTO, destination_city=MIAMI, value=10),
    Ticket(id=13, source_city=DULUTH, destination_city=EL_PASO, value=10),
    Ticket(id=14, source_city=PORTLAND, destination_city=PHOENIX, value=11),
    Ticket(id=15, source_city=DALLAS, destination_city=NEW_YORK, value=11),
    Ticket(id=16, source_city=DENVER, destination_city=PITTSBURGH, value=11),
    Ticket(id=17, source_city=WINNIPEG, destination_city=LITTLE_ROCK, value=11),
    Ticket(id=18, source_city=WINNIPEG, destination_city=HOUSTON, value=12),
    Ticket(id=19, source_city=BOSTON, destination_city=MIAMI, value=12),
    Ticket(id=20, source_city=CALGARY, destination_city=PHOENIX, value=13),
    Ticket(id=21, source_city=MONTREAL, destination_city=NEW_ORLEANS, value=13),
    Ticket(id=22, source_city=VANCOUVER, destination_city=SANTA_FE, value=13),
    Ticket(id=23, source_city=LOS_ANGELES, destination_city=CHICAGO, value=16),
    Ticket(id=24, source_city=PORTLAND, destination_city=NASHVILLE, value=17),
    Ticket(id=25, source_city=SAN_FRANCISCO, destination_city=ATLANTA, value=17),
    Ticket(id=26, source_city=VANCOUVER, destination_city=MONTREAL, value=20),
    Ticket(id=27, source_city=LOS_ANGELES, destination_city=MIAMI, value=20),
    Ticket(id=28, source_city=LOS_ANGELES, destination_city=NEW_YORK, value=21),
    Ticket(id=29, source_city=SEATTLE, destination_city=NEW_YORK, value=22),
]
