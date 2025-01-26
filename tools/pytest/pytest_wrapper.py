import sys

import pytest

if __name__ == "__main__":
    args = sys.argv[1:]

    targets = []
    filters = []
    remaining = []
    for arg in sys.argv[1:]:
        if arg.startswith("--target="):
            targets.append(arg.removeprefix("--target="))
        elif not arg.startswith("-") and ("::" in arg or arg.endswith(".py")):
            filters.append(arg)
        else:
            remaining.append(arg)

    if filters:
        targets = [f for f in filters if any(f.startswith(target) for target in targets)]

    assert len(targets) >= 1, f"There must be at least one target in arguments: {args}"

    pytest_args = targets + remaining

    sys.exit(pytest.main(pytest_args))
