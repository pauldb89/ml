from typing import Any
from typing import Dict
from typing import List
from typing import Protocol

from wordle.environment import State


class Policy(Protocol):
    def predict(self, guesses: State) -> str:
        ...
