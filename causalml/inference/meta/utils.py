import pandas as pd
import numpy as np

from packaging import version
from xgboost import __version__ as xgboost_version

# Optional Polars import
try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    pl = None
    _POLARS_AVAILABLE = False


def _is_polars_dataframe(obj):
    """Return True if *obj* is a polars DataFrame or LazyFrame."""
    if not _POLARS_AVAILABLE:
        return False
    return isinstance(obj, (pl.DataFrame, pl.LazyFrame))


def _is_polars_series(obj):
    """Return True if *obj* is a polars Series."""
    if not _POLARS_AVAILABLE:
        return False
    return isinstance(obj, pl.Series)


def _polars_to_numpy(obj):
    """Convert a polars DataFrame, LazyFrame, or Series to a NumPy array.

    - ``pl.LazyFrame`` is collected first (implicit ``.collect()``).
    - A single-column ``pl.DataFrame`` is squeezed to a 1-D array to match
      the behaviour of ``pd.Series.to_numpy()``.
    - A multi-column ``pl.DataFrame`` is returned as a 2-D array.
    """
    if isinstance(obj, pl.LazyFrame):
        obj = obj.collect()

    if isinstance(obj, pl.DataFrame):
        arr = obj.to_numpy()
        # Squeeze single-column frames so downstream code gets a 1-D vector
        if arr.shape[1] == 1:
            return arr.ravel()
        return arr

    if isinstance(obj, pl.Series):
        return obj.to_numpy()

    raise TypeError(f"Expected a polars DataFrame/LazyFrame/Series, got {type(obj)}")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def convert_pd_to_np(*args):
    """Convert pandas or polars objects to NumPy arrays.

    Accepts any mix of:
    * ``pd.DataFrame`` / ``pd.Series``   → ``.to_numpy()``
    * ``pl.DataFrame`` / ``pl.LazyFrame`` / ``pl.Series``  → converted via
      :func:`_polars_to_numpy`
    * Any other type (e.g. ``np.ndarray``, ``None``) → returned unchanged.
    """

    def _convert(obj):
        if obj is None:
            return obj
        if _POLARS_AVAILABLE and isinstance(
            obj, (pl.DataFrame, pl.LazyFrame, pl.Series)
        ):
            return _polars_to_numpy(obj)
        if hasattr(obj, "to_numpy"):
            return obj.to_numpy()
        return obj

    output = [_convert(obj) for obj in args]
    return output if len(output) > 1 else output[0]


def check_treatment_vector(treatment, control_name=None):
    n_unique_treatments = np.unique(treatment).shape[0]
    assert n_unique_treatments > 1, "Treatment vector must have at least two levels."
    if control_name is not None:
        assert (
            control_name in treatment
        ), "Control group level {} not found in treatment vector.".format(control_name)


def check_p_conditions(p, t_groups):
    eps = np.finfo(float).eps

    # Build the allowed types tuple dynamically so it works whether or not
    # polars is installed.
    _allowed = [np.ndarray, pd.Series]
    if _POLARS_AVAILABLE:
        _allowed.append(pl.Series)
    _allowed_tuple = tuple(_allowed)

    assert isinstance(
        p, (*_allowed_tuple, dict)
    ), "p must be an np.ndarray, pd.Series, pl.Series (if polars is installed), or dict type"

    if isinstance(p, _allowed_tuple):
        # Normalise to numpy for the value checks below
        p_np = p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)
        assert (
            t_groups.shape[0] == 1
        ), "If p is passed as an array/Series, there must be only 1 unique non-control group in the treatment vector."
        assert (0 + eps < p_np).all() and (
            p_np < 1 - eps
        ).all(), "The values of p should lie within the (0, 1) interval."

    if isinstance(p, dict):
        for t_name in t_groups:
            p_val = p[t_name]
            p_np = p_val.to_numpy() if hasattr(p_val, "to_numpy") else np.asarray(p_val)
            assert (0 + eps < p_np).all() and (
                p_np < 1 - eps
            ).all(), "The values of p should lie within the (0, 1) interval."


def check_explain_conditions(method, models, X=None, treatment=None, y=None):
    valid_methods = ["gini", "permutation", "shapley"]
    assert method in valid_methods, "Current supported methods: {}".format(
        ", ".join(valid_methods)
    )

    if method in ("gini", "shapley"):
        conds = [hasattr(mod, "feature_importances_") for mod in models]
        assert all(
            conds
        ), "Both models must have .feature_importances_ attribute if method = {}".format(
            method
        )

    if method in ("permutation", "shapley"):
        assert all(
            arr is not None for arr in (X, treatment, y)
        ), "X, treatment, and y must be provided if method = {}".format(method)


def clean_xgboost_objective(objective):
    """
    Translate objective to be compatible with loaded xgboost version

    Args
    ----

    objective : string
        The objective to translate.

    Returns
    -------
    The translated objective, or original if no translation was required.
    """
    compat_before_v83 = {"reg:squarederror": "reg:linear"}
    compat_v83_or_later = {"reg:linear": "reg:squarederror"}
    if version.parse(xgboost_version) < version.parse("0.83"):
        if objective in compat_before_v83:
            objective = compat_before_v83[objective]
    else:
        if objective in compat_v83_or_later:
            objective = compat_v83_or_later[objective]
    return objective


def get_xgboost_objective_metric(objective):
    """
    Get the xgboost version-compatible objective and evaluation metric from a potentially version-incompatible input.

    Args
    ----

    objective : string
        An xgboost objective that may be incompatible with the installed version.

    Returns
    -------
    A tuple with the translated objective and evaluation metric.
    """

    def clean_dict_keys(orig):
        return {clean_xgboost_objective(k): v for (k, v) in orig.items()}

    metric_mapping = clean_dict_keys(
        {"rank:pairwise": "auc", "reg:squarederror": "rmse"}
    )

    objective = clean_xgboost_objective(objective)

    assert (
        objective in metric_mapping
    ), "Effect learner objective must be one of: " + ", ".join(metric_mapping)
    return objective, metric_mapping[objective]


def get_weighted_variance(x, sample_weight):
    """
    Calculate the variance of array x with sample_weight.

    Args
    ----

    x : (np.array)
        A list of number

    sample_weight (np.array or list): an array of sample weights indicating the
        weight of each observation for `effect_learner`. If None, it assumes equal weight.

    Returns
    -------
    The variance of x with sample weight
    """
    average = np.average(x, weights=sample_weight)
    variance = np.average((x - average) ** 2, weights=sample_weight)
    return variance
