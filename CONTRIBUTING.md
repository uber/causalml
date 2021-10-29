# Contributing to Causal ML

The Causal ML project welcome community contributors.
To contribute to it, please follow guidelines here.

The codebase is hosted on Github at https://github.com/uber/causalml.

All code need to follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/) with a few exceptions listed in [tox.ini](./tox.ini).

Before contributing, please review outstanding issues.
If you'd like to contribute to something else, open an issue for discussion first.

# Recommended Development Workflow

1. Fork the `causalml` repo. This will create your own copy of the `causalml` repo. For more details about forks, please check [this guide](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/about-forks) at GitHub.
2. Clone the forked repo locally
3. Create a branch for the change:
```bash
$ git checkout -b branch_name
```
4. Make a change
5. Test your change as described below in the Test section
6. Commit the change to your local branch
```
$ git add file1_changed file2_changed
$ git commit -m "Issue number: message to describe the change."
```
7. Push your local branch to remote
```
$ git push origin branch_name
```
8. Go to GitHub and create PR from your branch in your forked repo to the original `causalml` repo. An instruction to create a PR from a fork is available [here](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)


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
