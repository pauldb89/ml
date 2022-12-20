def match(guessed_word: str, target_word: str) -> str:
    """
    Returns a value in [0, 3^5) encoding how well a guessed word matches the target word in base 3.

    When writing the output in base 3 a 0 means that the current letter doesn't exist in the target word, a 1 means
    that the letter exists in the target word, but not at the current position, while a 2 means that the letter and
    the position match.

    :param guessed_word: The guessed word.
    :param target_word: The hidden target word.
    :return: A state encoding how well the match word matches the guessed word.
    """
    target_letters = set(target_word)
    state = []
    for guess_letter, target_letter in zip(guessed_word, target_word):
        if guess_letter == target_letter:
            state.append("2")
        elif guess_letter in target_letters:
            state.append("1")
        else:
            state.append("0")

    return "".join(state)
