from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from gym import spaces
from gym.core import ActType
from gym.core import ObsType
from gym.core import RenderFrame

from wordle.consts import MAX_GUESSES
from wordle.consts import WORD_LENGTH
from wordle.spaces import VocabularySpace
from wordle.utils import match
from wordle.vocabulary import solution_vocabulary
from wordle.vocabulary import vocabulary


State = List[Tuple[str, str]]


class WordleEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # The observation space is a sequence of up to 6 entries, each entry consisting of a word from the dictionary
        # and one out of 3 colors (gray, green, yellow) for each of its 5 letters.
        self.observation_space = spaces.Sequence(
            spaces.Tuple([
                VocabularySpace(vocabulary=vocabulary),
                spaces.Discrete(3 ** WORD_LENGTH),
            ])
        )
        # The action space is the vocabulary (picking the next word).
        self.action_space = VocabularySpace(vocabulary=vocabulary)
        # The solution vocabulary which is a subset of the state vocabulary.
        self.solution_space = VocabularySpace(vocabulary=solution_vocabulary)

        self.render_mode = "rgb_array"

        self._reset()

    def _select_hidden_word(self) -> str:
        return self.solution_space.sample()

    def _reset(self, seed: Optional[int] = None) -> None:
        self.observation_space.seed(seed)
        self.solution_space.seed(seed)
        self.action_space.seed(seed)

        self.guesses: State = []
        self.hidden_word: str = self._select_hidden_word()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        self._reset(seed=seed)

        return self.guesses, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.action_space.contains(action)

        match_state = match(action, self.hidden_word)
        self.guesses.append((action, match_state))

        # Here we do simple reward shaping: 1 for guessing the hidden word, -1 for running out of guesses or 0 if some
        # guesses are left.
        done = len(self.guesses) >= MAX_GUESSES or action == self.hidden_word
        if action == self.hidden_word:
            reward = 100.0
        elif len(self.guesses) > MAX_GUESSES:
            reward = -100.0
        else:
            reward = 0.0

        return self.guesses, reward, done, False, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Andale Mono.ttf", 100)
        state_colors = {'0': 'grey', '1': 'yellow', '2': 'green'}
        padding = 10

        canvas = np.full(shape=(600, 500, 3), fill_value=100, dtype=np.uint8)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        for i, (word_guess, match_state) in enumerate(self.guesses):
            for j, (letter, letter_state) in enumerate(zip(word_guess, match_state)):
                position = (j * 100 + 20, i * 100)
                text = letter.upper()
                bbox = draw.textbbox(position, text=text, font=font, align="center")
                padded_bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
                draw.rectangle(padded_bbox, fill=state_colors[letter_state])
                draw.text(xy=position, text=text, font=font, color=(255, 255, 255), align="center")

        return img


if __name__ == "__main__":
    env = WordleEnv()
    env.reset(seed=0)
    print(env.hidden_word)
    for _ in range(5):
        word = env.action_space.sample()
        env.step(word)
        image = env.render()
        plt.imshow(image)
        plt.show()
