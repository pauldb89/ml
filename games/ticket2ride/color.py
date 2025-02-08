from dataclasses import dataclass

from termcolor import colored


@dataclass(frozen=True, order=True)
class Color:
    id: int
    name: str

    def __repr__(self) -> str:
        color_overrides: dict[str, str] = {
            "pink": "light_magenta",
            "orange": "light_red",
            "any": "magenta",
        }
        return f"{colored(self.name, color=color_overrides.get(self.name, self.name))}"


PINK = Color(id=0, name="pink")
WHITE = Color(id=1, name="white")
BLUE = Color(id=2, name="blue")
YELLOW = Color(id=3, name="yellow")
ORANGE = Color(id=4, name="orange")
BLACK = Color(id=5, name="black")
RED = Color(id=6, name="red")
GREEN = Color(id=7, name="green")
ANY = Color(id=8, name="any")

COLORS = [
    PINK,
    WHITE,
    BLUE,
    YELLOW,
    ORANGE,
    BLACK,
    RED,
    GREEN,
]

EXTENDED_COLORS = COLORS + [ANY]
