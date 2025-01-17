load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("//tools/pytest:defs.bzl", "pytest_test")

py_library(
    name = "board_games",
    srcs = glob(["**/*.py"], exclude = ["tests/**"]),
    deps = [
        "@pypi//tabulate",
        "@pypi//termcolor",
    ]
)

pytest_test(
    name = "board_games_test",
    srcs = glob(["tests/**/test*.py"]),
    deps = [":board_games"],
)
