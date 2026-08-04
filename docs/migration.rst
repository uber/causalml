=================
Migration Guide
=================

.. _fit-argument-order:

``fit()`` argument order changes in v1.0
========================================

**If you only read one thing:** pass ``treatment`` and ``y`` as keyword
arguments, starting now.

.. code-block:: python

    # Before — breaks silently at v1.0
    learner.fit(X, treatment, y)

    # After — works identically before and after v1.0
    learner.fit(X=X, treatment=treatment, y=y)

That single edit is the whole migration. Keyword arguments are
order-independent, so a call written this way behaves the same in every version
and needs no second edit when the flip lands.

Why this is changing
--------------------

CausalML learners have always taken ``fit(X, treatment, y, ...)``, which puts
``y`` third. ``sklearn.pipeline.Pipeline`` calls its final estimator's
``fit(X, y)`` positionally, so a CausalML learner cannot be a pipeline step
without an adapter — the problem reported in `#854
<https://github.com/uber/causalml/issues/854>`_.

In v1.0 the positional order becomes ``fit(X, y, treatment, ...)``, matching the
scikit-learn convention.

.. warning::

    This change cannot be detected at runtime. ``y`` and ``treatment`` are both
    same-length arrays, so a call left in the old positional form will **not**
    raise ``TypeError`` after the flip — it will train on swapped arguments and
    return plausible-looking, wrong numbers.

    This is why the deprecation warns for a full release cycle before anything
    moves, and why keyword arguments are the recommended fix rather than
    "reorder your positional calls at v1.0".

What happens when
-----------------

**0.18.0 (September 2026), the release that ships this guide.** Positional order
is *unchanged*. Passing ``treatment`` or ``y`` positionally emits a
``FutureWarning`` pointing here. Nothing breaks.

**0.19.0 (December 2026) and 0.20.0 (March 2027).** Unchanged — the warning
stays, the signatures do not move.

**v1.0 (June 2027).** The signatures are reordered and the warning is removed.
Positional calls in the old order start silently mis-training; keyword calls are
unaffected.

That is **three minor releases and roughly nine months** of warning before
anything moves. If that is not enough for your codebase, say so on the `v1.0
roadmap discussion <https://github.com/uber/causalml/discussions/938>`_.

Release dates are targets rather than commitments; the *ordering* is the part
you can rely on, and no signature moves before v1.0.

The rule
--------

One rule, applied uniformly across the package:

    **X, then** ``y``, **then** ``treatment``\ **, and every other parameter
    keeps its relative position.**

So the two arguments move to the front in scikit-learn order, and everything
else — ``p``, ``sample_weight``, ``return_ci``, ``verbose`` — shifts right
without being reshuffled among itself.

The rule is implemented as ``v1_order()`` in ``causalml/inference/_arg_order.py``
and derived per method from that method's own signature, so the target for every
method is machine-checkable rather than hand-maintained.

Finding the calls you need to change
------------------------------------

Run your test suite or a representative script with the warning made fatal, so
each call site surfaces as a traceback:

.. code-block:: bash

    python -W error::FutureWarning your_script.py

That also promotes unrelated ``FutureWarning``\ s from pandas, NumPy and
scikit-learn, so on a large codebase it is usually easier to collect only this
deprecation and keep going:

.. code-block:: python

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ...  # your code
    for w in caught:
        if issubclass(w.category, FutureWarning) and "argument order" in str(w.message):
            print(f"{w.filename}:{w.lineno}")

The warning is raised with ``stacklevel=2``, so ``filename`` and ``lineno``
point at *your* call site, not at CausalML internals.

A grep is less reliable than it looks: ``fit(X, a, b)`` is also how you call a
plain scikit-learn estimator, and those are unaffected. Prefer the warning.

What changes, by family
-----------------------

Every public method whose positional order changes is listed below. ``predict``
is included deliberately: its current order does not block ``Pipeline``, since
``predict(X)`` already works, but flipping only ``fit`` would leave one class
with the same two arguments in opposite positional orders on two methods.

Meta-learners
~~~~~~~~~~~~~

``causalml.inference.meta`` — the S/T/X/R/DR learners and their classifier
variants, ``XGBTRegressor``, ``XGBRRegressor``, ``XGBTClassifier``,
``XGBRClassifier``, ``LRSRegressor``, ``TMLELearner``.

Methods: ``fit``, ``predict``, ``fit_predict``, ``estimate_ate``, ``bootstrap``,
``fit_bootstrap_ensemble``.

.. code-block:: text

    now:  (X, treatment, y, p, ...)
    v1.0: (X, y, treatment, p, ...)

Causal trees and forests
~~~~~~~~~~~~~~~~~~~~~~~~

``CausalTreeRegressor``, ``CausalRandomForestRegressor``.

Methods: ``fit``, ``fit_predict``, ``estimate_ate``, ``bootstrap``,
``bootstrap_pool``.

.. code-block:: text

    now:  (X, treatment, y, sample_weight, ...)
    v1.0: (X, y, treatment, sample_weight, ...)

Uplift trees and forests
~~~~~~~~~~~~~~~~~~~~~~~~

``UpliftTreeClassifier``, ``UpliftRandomForestClassifier``.

Methods: ``fit``, ``fill``, ``prune``.

``UpliftTreeClassifier.fit`` also takes a validation triple, which is reordered
in place so it stays consistent with the main one:

.. code-block:: text

    now:  (X, treatment, y, X_val, treatment_val, y_val, sample_weight, check_input)
    v1.0: (X, y, treatment, X_val, y_val, treatment_val, sample_weight, check_input)

Instrumental variables
~~~~~~~~~~~~~~~~~~~~~~

Two signatures here are **not** a plain ``treatment``/``y`` swap.

``IVRegressor.fit`` — the instrument ``w`` keeps its relative position and
shifts right:

.. code-block:: text

    now:  (X, treatment, y, w)
    v1.0: (X, y, treatment, w)

``BaseDRIVLearner`` (``fit``, ``fit_predict``, ``estimate_ate``, ``bootstrap``)
— ``assignment`` currently sits *second*, and lands fourth:

.. code-block:: text

    now:  (X, assignment, treatment, y, p, pZ, ...)
    v1.0: (X, y, treatment, assignment, p, pZ, ...)

``BaseDRIVLearner.predict`` takes no ``assignment`` and follows the ordinary
rule: ``(X, treatment, y, ...)`` becomes ``(X, y, treatment, ...)``.

Neural estimators
~~~~~~~~~~~~~~~~~

``DragonNet`` (TensorFlow and JAX backends) and ``CEVAE`` (PyTorch and JAX
backends). Methods: ``fit``, ``predict``, ``fit_predict``.

.. code-block:: text

    now:  (X, treatment, y, p)
    v1.0: (X, y, treatment, p)

Optional backends are only importable with the matching extra installed, so
these calls may not warn in an environment where the backend is missing. They
change all the same.

Policy learning
~~~~~~~~~~~~~~~

``PolicyLearner.fit``:

.. code-block:: text

    now:  (X, treatment, y, p, dhat)
    v1.0: (X, y, treatment, p, dhat)

Sensitivity analysis helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``causalml.metrics.sensitivity.Sensitivity`` — ``get_prediction``,
``get_ate_ci``, ``get_potential_outcome_predictions``. These are a third shape,
with ``p`` currently second; it shifts right:

.. code-block:: text

    now:  (X, p, treatment, y)
    v1.0: (X, y, treatment, p)

``Sensitivity.summary`` and ``sensitivity_analysis`` read ``treatment`` and
``y`` from a DataFrame by column name and are unaffected.

Deliberately unchanged
----------------------

A method is in scope only if **both** ``y`` and ``treatment`` appear after the
first position. Anything else keeps its current signature at v1.0:

* ``BaseRLearner.predict(X, p, return_components)`` — takes neither, so there is
  nothing to reorder. After the flip, ``BaseRLearner.fit`` and ``.predict``
  legitimately differ; that is not an oversight.
* ``ElasticNetPropensityModel`` and the other ``PropensityModel`` classes —
  ``fit(X, y)`` and ``fit_predict(X, y)`` take no ``treatment`` and already
  follow the scikit-learn convention.
* Dataset helpers that name their treatment vector something other than
  ``treatment``, such as ``SemiSynthDataGenerator.fit(X, w, y)``.
* Any plain scikit-learn estimator you pass *into* a CausalML learner. Only
  CausalML's own signatures change.

If a method takes only ``y`` and no ``treatment``, this deprecation does not
apply to it.
