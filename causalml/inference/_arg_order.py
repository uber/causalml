"""#854: scikit-learn ``fit(X, y, ...)`` argument-order migration.

CausalML learners historically take ``fit(X, treatment, y, ...)``, which puts
``y`` third and breaks ``sklearn.pipeline.Pipeline`` (it calls the final
estimator's ``fit(X, y)`` positionally). In v1.0 the positional order becomes
``(X, y, treatment, ...)``; every other parameter keeps its relative position,
so ``BaseDRIVLearner.fit(X, assignment, treatment, y, ...)`` becomes
``fit(X, y, treatment, assignment, ...)``.

This is a two-step deprecation. Step one (this module): the positional order is
UNCHANGED, so no existing call silently breaks, but passing ``treatment`` or
``y`` positionally emits a ``FutureWarning`` steering callers to keyword
arguments, which are order-independent and therefore safe across the flip.
Step two (v1.0): reorder the signatures and delete this module.
"""

import functools
import inspect
import warnings

#: Parameters whose positional slot changes in v1.0, in their v1.0 order.
REORDERED_PARAMS = ("y", "treatment")

#: Methods wrapped automatically for ``SerializableLearner`` subclasses.
SHIMMED_METHODS = ("fit", "fit_predict", "estimate_ate", "predict")

_MSG = (
    "Passing `treatment` and/or `y` to {name}() by position is deprecated and "
    "will change in causalml v1.0: the positional argument order becomes "
    "({new_order}) for scikit-learn Pipeline compatibility "
    "(see https://github.com/uber/causalml/issues/854). To be safe across the "
    "change, pass them as keyword arguments, e.g. {example}."
)


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


def v1_order(params):
    """Reorder ``params`` to the v1.0 convention: X, then y, treatment, then rest.

    Parameters other than ``y``/``treatment`` keep their relative order, so a
    signature's extra arguments (``assignment``, ``p``, ``sample_weight``, ...)
    simply shift right.
    """
    head, tail = params[:1], params[1:]
    moved = [name for name in REORDERED_PARAMS if name in tail]
    return head + moved + [name for name in tail if name not in REORDERED_PARAMS]


def deprecate_positional_treatment_y(method):
    """Warn when ``treatment``/``y`` are passed positionally to ``method``.

    ``method`` is returned unchanged when neither parameter is positional, so
    this is safe to apply blanket-style from ``__init_subclass__``.

    Only the outermost shimmed call on a given instance warns, so internal
    delegation (``fit_predict`` -> ``fit``/``predict``, a subclass ``fit`` ->
    ``super().fit``, ``estimate_ate`` -> ``fit_predict`` -> ``fit``) never
    double-counts and callers see at most one warning per top-level call.
    :func:`functools.wraps` preserves the real signature, so introspection and
    scikit-learn metadata routing are unaffected.
    """
    params = _positional_params(method)
    if not any(name in REORDERED_PARAMS for name in params):
        return method

    message = _MSG.format(
        name=method.__name__,
        new_order=", ".join(v1_order(params)),
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
    """Wrap ``cls``'s own :data:`SHIMMED_METHODS` with the arg-order shim.

    Only methods defined directly on ``cls`` are wrapped (inherited ones are
    already wrapped on the parent), abstract methods are left alone, and
    already-wrapped methods are skipped, so nothing is double-wrapped.

    ``SerializableLearner`` applies this automatically via ``__init_subclass__``.
    Learners outside that hierarchy — the TF/Torch/JAX estimators,
    ``PolicyLearner``, ``TMLELearner`` — use it as a class decorator.
    """
    for name in SHIMMED_METHODS:
        method = cls.__dict__.get(name)
        if (
            callable(method)
            and not getattr(method, "_arg_order_shimmed", False)
            and not getattr(method, "__isabstractmethod__", False)
        ):
            setattr(cls, name, deprecate_positional_treatment_y(method))
    return cls
