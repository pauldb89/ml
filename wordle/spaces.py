from typing import Any
from typing import Optional
from typing import Union

import gym
import numpy as np

from wordle.vocabulary import Vocabulary


class VocabularySpace(gym.Space[str]):
    def __init__(self, vocabulary: Vocabulary, seed: Optional[Union[int, np.random.Generator]] = None):
        super().__init__((), np.int64, seed)
        self.vocabulary = vocabulary

    @property
    def is_np_flattenable(self):
        return False

    def sample(self, mask: Optional[Any] = None) -> str:
        assert mask is None, "Sampling with a mask is not supported"

        word_id = self.np_random.integers(low=0, high=len(self.vocabulary)-1)
        return self.vocabulary.get_word(word_id)

    def contains(self, x: str) -> bool:
        return x == "" or x in self.vocabulary
