# Contributing to CausalML

The **CausalML** project welcome community contributors.
To contribute to it, please follow guidelines here.

The codebase is hosted on Github at https://github.com/uber/causalml.

We use [`black`](https://black.readthedocs.io/en/stable/index.html) as a formatter to keep the coding style and format across all Python files consistent and compliant with [PEP8](https://www.python.org/dev/peps/pep-0008/). We recommend that you add `black` to your IDE as a formatter (see the [instruction](https://black.readthedocs.io/en/stable/integrations/editors.html)) or run `black` on the command line before submitting a PR as follows:
```bash
# move to the top directory of the causalml repository
$ cd causalml
$ pip install -U black
$ black .
```

Additionally, you can set up black and other tools we use to run before any commit is made via:
```bash
make setup_local
```

As a start, please check out outstanding [issues](https://github.com/uber/causalml/issues).
If you'd like to contribute to something else, open a new issue for discussion first.

## Development Workflow :computer:

1. Fork the `causalml` repo. This will create your own copy of the `causalml` repo. For more details about forks, please check [this guide](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/about-forks) at GitHub.
2. Clone the forked repo locally
3. (optional) Complete local installation by running:
```bash
make setup_local
```
4. Create a branch for the change:
```bash
$ git checkout -b branch_name
```
5. Make a change
6. Test your change as described below in the Test section
7. Commit the change to your local branch
```bash
$ git add file1_changed file2_changed
$ git commit -m "Issue number: message to describe the change."
```
8. Push your local branch to remote
```bash
$ git push origin branch_name
```
9. Go to GitHub and create PR from your branch in your forked repo to the original `causalml` repo. An instruction to create a PR from a fork is available [here](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

## Documentation :books:

[**CausalML** documentation](https://causalml.readthedocs.io/) is generated with [Sphinx](https://www.sphinx-doc.org/en/master/) and hosted on [Read the Docs](https://readthedocs.org/).

### Docstrings

All public classes and functions should have docstrings to specify their inputs, outputs, behaviors and/or examples. For docstring conventions in Python, please refer to [PEP257](https://www.python.org/dev/peps/pep-0257/).

**CausalML** supports the NumPy and Google style docstrings in addition to Python's original docstring with [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html). Google style docstrings are recommended for simplicity. You can find examples of Google style docstrings [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

### Generating Documentation Locally

You can generate documentation in HTML locally as follows:
```bash
$ cd docs/
$ pip install -r requirements.txt
$ make html
```

Documentation will be available in `docs/_build/html/index.html`.

## Test :wrench:

If you added a new inference method, add test code to the `tests/` folder.

### Prerequisites

**CausalML** uses `pytest` for tests. Install `pytest` and `pytest-cov`, and the package dependencies:
```bash
$ pip install .[test]
```
See details for test dependencies in `pyproject.toml`

### Building Cython

In order to run tests, you need to build the Cython modules
```bash
$ python setup.py build_ext --inplace
```
This is important because during testing causalml modules are imported from the source code.

### Testing

Before submitting a PR, make sure the change to pass all tests and test coverage to be at least 70%.
```bash
$ pytest -vs tests/ --cov causalml/
```

To run tests that require tensorflow (i.e. DragonNet), make sure tensorflow is installed and include the `--runtf` option with the `pytest` command.  For example:

```bash
$ pytest --runtf -vs tests/test_dragonnet.py
```

You can also run tests via make:
```bash
$ make test
```



## Submission :tada:

In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies
- References

Please add the core Causal ML contributors as reviewers.

## Maintain in `conda-forge`  :snake:

We are supporting to install the package through `conda`, in order to maintain the packages in conda we need to keep the package's version in conda's recipe repository [here](https://github.com/conda-forge/causalml-feedstock) in sync with `CausalML`. You can follow the [instruction](https://conda-forge.org/#update_recipe) from conda or below steps:

1. After a new release of the package, fork the repo.
2. Create a new branch from the master branch.
3. Edit the recipe:
    - Update the version number [here](https://github.com/conda-forge/causalml-feedstock/blob/main/recipe/meta.yaml#L2) in `meta.yaml`
    - Generate the new sha256 hash and update it [here](https://github.com/conda-forge/causalml-feedstock/blob/main/recipe/meta.yaml#L11):  the sha256 hash can get from PyPi; look for the SHA256 link next to the download link on PyPi package’s files page, e.g. https://pypi.org/project/causalml/#files
    - Reset the build number to 0
    - Update the dependencies if needed
4. Submit the PR and the recipe will automatically be built;

Once the recipe is ready it will be merged. The recipe will then automatically be built and uploaded to the conda-forge channel.
