# Contributing to Causal ML

The Causal ML project welcome community contributors.
To contribute to it, please follow guidelines here.

The codebase is hosted on Github at https://github.com/uber/causalml.

All code need to follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).

Before contributing, please review outstanding issues.
If you'd like to contribute to something else, open an issue for discussion first.


# Test

Causal ML uses `pytest` for tests.
If you added a new inference method, add test code to the `tests/` folder.

Before submitting a PR, make sure the change to pass all tests.
```sh
pytest -vs
```


# Submission

In your PR, please include:
- Changes made
- Links to related issues/PRs
- Dependencies
- References

Please add the core Causal ML contributors as reviewers.