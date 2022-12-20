import json
from typing import List
from typing import Tuple

import requests

from wordle.consts import WORD_LENGTH
from wordle.environment import State


class ExpertPolicy:
    URL = "https://wordledictionary.com/.netlify/functions/query"

    def predict(self, guesses: State) -> str:
        exact_matches = ["_"] * WORD_LENGTH
        matching_letters = set()
        excluded_letters = set()

        for word_guess, match_state in guesses:
            for idx, (letter, letter_state) in enumerate(zip(word_guess, match_state)):
                if letter_state == "0":
                    excluded_letters.add(letter)
                elif letter_state == "1":
                    matching_letters.add(letter)
                else:
                    assert letter_state == "2"
                    exact_matches[idx] = letter

        params = {
            "find": "".join(exact_matches),
            "has": ",".join(matching_letters),
            "not": ",".join(excluded_letters),
        }

        response = requests.get(url=self.URL, params=params)
        assert response.status_code == 200, "The expert policy needs the HTTP request to work"

        json_response = json.loads(response.text)
        results = json_response["results"]
        assert len(results) > 0, "No valid results for the given state"

        result = min(results, key=lambda record: (-record["score"], record["word"]))
        return result["word"]


def main():
    policy = ExpertPolicy()
    print(policy.predict([]))
    print(policy.predict([("arise", "12100")]))
    print(policy.predict([("arise", "12100"), ("train", "02220")]))
    print(policy.predict([("arise", "12100"), ("train", "02220"), ("frail", "02220")]))


if __name__ == "__main__":
    main()
