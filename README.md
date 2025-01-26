Toy ML projects.

### Bazel

These instructions assume `bazel 8.0.1` is installed.

`bazel_requirements.in` lists the packages on which we have actual dependencies.
The transitive list of dependencies is generated in `bazel_requirements.txt` with:
```shell
pip-compile bazel_requirements.in --output-file=bazel_requirements.txt --allow-unsafe
```

Run the test for a package with 
```shell
bazel test //board_games:board_games_test
```

or individual tests with 
```shell
bazel test //board_games:board_games_test --test_arg=board_games/tests/ticket2ride/test_policy_helpers.py::test_get_valid_actions
```

### Weights and Biases

Set the `WANDB_ENTITY` and `WANDB_API_KEY` environment variables in `~/.zshrc`.
