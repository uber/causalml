============
Contributing
============

CausalML is developed in the open at `github.com/uber/causalml
<https://github.com/uber/causalml>`_, and contributions are welcome — bug
reports, documentation, new estimators, and reviews of open pull requests.

Setting up a development environment
------------------------------------

Follow :ref:`Install from source <installation:Install from source>`. The
editable install compiles the package's Cython extensions, so no separate build
step is needed. Then run the test suite as described in
:ref:`Running Tests <installation:Running Tests>`.

After editing any ``.pyx`` or ``.pxd`` file, reinstall so the extension is
rebuilt::

    pip install -e . --no-deps

Before opening a pull request
-----------------------------

* Format the diff with `black <https://black.readthedocs.io/>`_. It is the
  formatting gate CI enforces.
* Add or update tests covering the behavior you changed.
* Add an entry under ``Unreleased`` in :doc:`changelog` if the change is
  user-visible — a new feature, a bug fix, or a change in behavior.
* If you change an estimator's signature, check
  :doc:`migration` — the ``fit(X, y, treatment, ...)`` transition is in
  progress and new methods are expected to follow the target order.

Project documents
-----------------

* `Contributing guide <https://github.com/uber/causalml/blob/master/CONTRIBUTING.md>`_
* `Code of conduct <https://github.com/uber/causalml/blob/master/CODE_OF_CONDUCT.md>`_
* `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_
* `Governance <https://github.com/uber/causalml/blob/master/GOVERNANCE.md>`_
* `Maintainers <https://github.com/uber/causalml/blob/master/MAINTAINERS.md>`_
* `Steering committee <https://github.com/uber/causalml/blob/master/STEERING_COMMITTEE.md>`_
* `Security policy <https://github.com/uber/causalml/blob/master/SECURITY.md>`_
