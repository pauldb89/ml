load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, deps = [], args = [], **kwargs):
    """
        Call pytest
    """
    py_test(
        name = name,
        srcs = [
            "//tools/pytest:pytest_wrapper.py",
        ] + srcs,
        main = "//tools/pytest:pytest_wrapper.py",
        args = [
            "--capture=no",
        ] + args + ["--target=$(locations :%s)" % x for x in srcs],
        python_version = "PY3",
        srcs_version = "PY3",
        deps = deps + ["@pypi//pytest"],
        imports = ["."],
        **kwargs
    )

