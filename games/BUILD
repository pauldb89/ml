load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("//tools/pytest:defs.bzl", "pytest_test")

py_library(
    name = "board_games",
    srcs = glob(["**/*.py"], exclude = ["tests/**", "**/scripts/**"]),
    deps = [
        "@pypi//numpy",
        "@pypi//tabulate",
        "@pypi//termcolor",
        "@pypi//torch",
    ]
)

py_binary(
    name = "simulate",
    srcs = ["ticket2ride/scripts/simulate.py"],
    main = "ticket2ride/scripts/simulate.py",
    deps = [":board_games"]
)

py_binary(
    name = "train",
    srcs = ["ticket2ride/scripts/train.py"],
    main = "ticket2ride/scripts/train.py",
    deps = [
        ":board_games",
        "@pypi//wandb",
    ]
)

pytest_test(
    name = "board_games_test",
    srcs = glob(["tests/**/test*.py"]),
    deps = [":board_games"],
)
