"""#854: scikit-learn ``fit(X, y, ...)`` argument-order migration.

CausalML learners historically take ``fit(X, treatment, y, ...)``, which puts
``y`` third and breaks ``sklearn.pipeline.Pipeline`` (it calls the final
estimator's ``fit(X, y)`` positionally). In v1.0 the positional order becomes
``(X, y, treatment, ...)``:

* ``y`` and ``treatment`` move to the front, immediately after ``X``, and every
  other parameter keeps its relative position — so
  ``BaseDRIVLearner.fit(X, assignment, treatment, y, ...)`` becomes
  ``fit(X, y, treatment, assignment, ...)``.
* A suffixed pair such as ``treatment_val``/``y_val`` is reordered in place, so
  ``UpliftTreeClassifier.fit(X, treatment, y, X_val, treatment_val, y_val, ...)``
  becomes ``fit(X, y, treatment, X_val, y_val, treatment_val, ...)`` rather than
  leaving the validation triple inconsistent with the main one.

This is a two-step deprecation. Step one (this module): the positional order is
UNCHANGED, so no existing call silently breaks, but passing ``treatment`` or
``y`` positionally emits a ``FutureWarning`` steering callers to keyword
arguments, which are order-independent and therefore safe across the flip.
Step two (v1.0): reorder the signatures and delete this module.

Coverage is decided by **signature, not by method name**. Any method whose
positional order would change is shimmed — ``fit`` and ``predict``, but equally
``bootstrap``, ``fit_bootstrap_ensemble``, ``prune`` and the ``Sensitivity``
helpers. An earlier name-based allowlist silently missed eleven public methods
(#981); the deprecation window is one-shot, so a method that is not shimmed here
would need a second deprecation cycle of its own.
"""

import functools
import inspect
import warnings

#: Parameters whose positional slot changes in v1.0, in their v1.0 order.
REORDERED_PARAMS = ("y", "treatment")


def _positional_params(method):
    """Return the names of ``method``'s positional parameters, minus ``self``."""
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    names = [
        p.name for p in inspect.signature(method).parameters.values() if p.kind in kinds
    ]
    return names[1:] if names[:1] == ["self"] else names


def _order_suffixed_pairs(names):
    """Put ``y`` before ``treatment`` within each suffixed pair, in place.

    Only pairs sharing a suffix are touched (``treatment_val``/``y_val``), and
    they keep the two slots they already occupy — unlike the bare
    ``y``/``treatment`` pair, which is hoisted to the front by :func:`v1_order`.
    """
    ordered = list(names)
    suffixes = {
        name[len("treatment") :]
        for name in ordered
        if name.startswith("treatment") and name != "treatment"
    }
    for suffix in suffixes:
        pair = ("y" + suffix, "treatment" + suffix)
        if all(name in ordered for name in pair):
            first, second = sorted(ordered.index(name) for name in pair)
            ordered[first], ordered[second] = pair
    return ordered


def v1_order(params):
    """Reorder ``params`` to the v1.0 convention.

    The feature matrix stays first, then ``y`` and ``treatment``, then
    everything else in its original relative order with any suffixed
    ``treatment``/``y`` pair swapped so ``y`` leads.

    ``params`` is returned unchanged unless **both** ``y`` and ``treatment``
    appear after the first position. This deprecation is about that pair; a
    method taking only one of them is not part of it, which keeps the rule off
    the vendored sklearn tree builders (``build(tree, X, y, ...)``, where
    hoisting ``y`` would put it ahead of ``X``) and off dataset helpers that
    name their treatment vector something else.
    """
    head, tail = params[:1], params[1:]
    if not all(name in tail for name in REORDERED_PARAMS):
        return list(params)
    moved = list(REORDERED_PARAMS)
    rest = [name for name in tail if name not in REORDERED_PARAMS]
    return head + moved + _order_suffixed_pairs(rest)


def _may_reorder(method):
    """Cheap pre-filter: could this method's positional order change at all?

    Reads the code object directly so the common case costs no
    :func:`inspect.signature` call — ``shim_arg_order`` scans every method of
    every learner class at import time. Unwraps first, because
    :func:`functools.wraps` copies ``__wrapped__`` but not ``__code__``, so a
    decorated method (e.g. ``@timeit``) would otherwise look argument-less.
    """
    code = getattr(inspect.unwrap(method), "__code__", None)
    if code is None:
        return False
    args = set(code.co_varnames[: code.co_argcount])
    return bool(args & set(REORDERED_PARAMS))


def deprecate_positional_treatment_y(method):
    """Warn when ``treatment``/``y`` are passed positionally to ``method``.

    ``method`` is returned unchanged when its positional order is already the
    v1.0 order (including when it takes neither parameter), so this is safe to
    apply blanket-style from ``__init_subclass__``.

    Only the outermost shimmed call on a given instance warns, so internal
    delegation (``fit_predict`` -> ``fit``/``predict``, a subclass ``fit`` ->
    ``super().fit``, ``estimate_ate`` -> ``fit_predict`` -> ``fit``) never
    double-counts and callers see at most one warning per top-level call.
    :func:`functools.wraps` preserves the real signature, so introspection and
    scikit-learn metadata routing are unaffected.
    """
    params = _positional_params(method)
    target = v1_order(params)
    if target == params:
        return method

    message = (
        "Passing `treatment` and/or `y` to {name}() by position is deprecated "
        "and will change in causalml v1.0: the positional argument order "
        "becomes ({new_order}) for scikit-learn Pipeline compatibility "
        "(see https://github.com/uber/causalml/issues/854). To be safe across "
        "the change, pass them as keyword arguments, e.g. {example}."
    ).format(
        name=method.__name__,
        new_order=", ".join(target),
        example="{}({}, {})".format(
            method.__name__,
            params[0],
            ", ".join(f"{p}={p}" for p in REORDERED_PARAMS if p in params),
        ),
    )

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_in_arg_order_call", False):
            return method(self, *args, **kwargs)
        if any(
            params[i] in REORDERED_PARAMS for i in range(min(len(args), len(params)))
        ):
            warnings.warn(message, FutureWarning, stacklevel=2)
        self._in_arg_order_call = True
        try:
            return method(self, *args, **kwargs)
        finally:
            self._in_arg_order_call = False

    wrapper._arg_order_shimmed = True
    return wrapper


def shim_arg_order(cls):
    """Shim every method of ``cls`` whose positional order changes in v1.0.

    Membership is decided by signature rather than by a list of method names —
    a name list is what let ``bootstrap``, ``prune`` and the ``Sensitivity``
    helpers slip through the window (#981). Only methods defined directly on
    ``cls`` are considered (inherited ones are already wrapped on the parent);
    dunders, ``classmethod``/``staticmethod`` descriptors, abstract methods and
    already-wrapped methods are skipped.

    ``SerializableLearner`` applies this automatically via ``__init_subclass__``.
    Classes outside that hierarchy — the TF/Torch/JAX estimators,
    ``PolicyLearner``, ``TMLELearner``, ``Sensitivity`` — use it as a decorator.
    """
    for name, method in list(vars(cls).items()):
        if (
            name.startswith("_")
            or isinstance(method, (classmethod, staticmethod))
            or not callable(method)
            or getattr(method, "__isabstractmethod__", False)
            or getattr(method, "_arg_order_shimmed", False)
            or not _may_reorder(method)
        ):
            continue
        setattr(cls, name, deprecate_positional_treatment_y(method))
    return cls
