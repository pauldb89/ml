from wordle.consts import WORD_LENGTH


class Vocabulary:
    def __init__(self, filename: str):
        self.words = []
        self.word_ids = {}

        for line in open(filename):
            word = line.strip()
            if not word:
                continue
            assert len(word) == WORD_LENGTH, f"{word} has invalid length {WORD_LENGTH}"

            self.words.append(word)
            self.word_ids[word] = len(self.word_ids)

    def get_word(self, word_id: int) -> str:
        return self.words[word_id]

    def get_word_id(self, word: str) -> int:
        return self.word_ids[word]

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.word_ids


vocabulary = Vocabulary("words.txt")
solution_vocabulary = Vocabulary("solution_words.txt")
