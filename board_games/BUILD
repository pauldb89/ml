load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("//tools/pytest:defs.bzl", "pytest_test")

py_library(
    name = "board_games",
    srcs = glob(["**/*.py"], exclude = ["tests/**"]),
    deps = [
        "@pypi//numpy",
        "@pypi//tabulate",
        "@pypi//termcolor",
        "@pypi//torch",
    ]
)

py_binary(
    name = "simulate",
    srcs = ["ticket2ride/simulate.py"],
    main = "ticket2ride/simulate.py",
    deps = [":board_games"]
)

pytest_test(
    name = "board_games_test",
    srcs = glob(["tests/**/test*.py"]),
    deps = [":board_games"],
)
