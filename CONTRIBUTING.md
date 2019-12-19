# Contributing to Causal ML

The Causal ML project welcome community contributors.
To contribute to it, please follow guidelines here.

The codebase is hosted on Github at https://github.com/uber/causalml.

All code need to follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/) with a few exceptions listed in [tox.ini](./tox.ini).

Before contributing, please review outstanding issues.
If you'd like to contribute to something else, open an issue for discussion first.


# Test

If you added a new inference method, add test code to the `tests/` folder.

## Prerequisites

Causal ML uses `pytest` for tests. Install `pytest` and `pytest-cov`, and the package dependencies:
```bash
$ pip install pytest pytest-cov -r requirements.txt
```

## Building Cython

In order to run tests, you need to build the Cython modules
```bash
$ python setup.py build_ext --inplace
```

## Testing

Before submitting a PR, make sure the change to pass all tests and test coverage to be at least 70%.
```bash
$ pytest -vs tests/ --cov causalml/
```


# Submission

In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies
- References

Please add the core Causal ML contributors as reviewers.