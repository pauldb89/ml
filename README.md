Toy ML projects.

### Remote development

We are currently using a linux based remote environment with all the cuda dependencies installed.

Create a virtual environment:

```shell
mkdir ~/code/.venv
virtualenv ~/code/.venv/ml
source ~/code/.venv/ml/bin/activate
pip install -r bazel_requirements.txt  # TODO(pauldb): Update this.
```

To run tests:
```shell
cd ~/code/ml
PYTHONPATH=. pytest board_games/tests 
```

To run scripts:
```shell
PYTHONPATH=. python board_games/ticket2ride/scripts/simulate.py
```

### Bazel

TODO(pauldb): Get rid of this -- using bazel is annoying as fuck.

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
