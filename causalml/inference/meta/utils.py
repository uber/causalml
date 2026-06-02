import pandas as pd
import numpy as np

from packaging import version
from xgboost import __version__ as xgboost_version

# ---------------------------------------------------------------------------
# Optional Polars import
# ---------------------------------------------------------------------------
try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    pl = None
    _POLARS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Native DataFrame helpers
# ---------------------------------------------------------------------------


def filter_mask(obj, mask):
    """Filter rows by a boolean mask.

    Works natively for numpy arrays, pandas DataFrames/Series, and
    polars DataFrames/Series without materialising unnecessary copies.

    Args:
        obj: numpy array, pd.DataFrame, pd.Series, pl.DataFrame, or pl.Series.
        mask: boolean array or Series of the same length as obj.

    Returns:
        Filtered object of the same type as input.
    """
    if obj is None:
        return None
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(mask, pd.Series):
            return obj.loc[mask]
        # numpy boolean array — reset index to avoid alignment issues
        return obj.loc[np.asarray(mask, dtype=bool)]
    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series)):
        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()
        if isinstance(mask, pl.Series):
            return obj.filter(mask)
        return obj.filter(pl.Series(np.asarray(mask, dtype=bool)))
    # numpy / anything else
    return obj[np.asarray(mask, dtype=bool)]


def filter_index(obj, indices):
    """Filter rows by integer positional indices.

    Used for KFold partition slicing (e.g. in DR-learner).

    Args:
        obj: numpy array, pd.DataFrame, pd.Series, pl.DataFrame, or pl.Series.
        indices: integer array of row positions.

    Returns:
        Filtered object of the same type as input.
    """
    if obj is None:
        return None
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[indices]
    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series)):
        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()
        return obj[indices]
    return obj[indices]  # numpy


def prepend_column(value, X):
    """Prepend a constant-value column to a feature matrix X.

    Used by the S-learner to prepend the treatment indicator before fitting.

    Args:
        value (float): scalar value to fill the new column with (0.0 or 1.0).
        X: numpy array, pd.DataFrame, or pl.DataFrame.

    Returns:
        Feature matrix with the new column prepended, same type as X.
    """
    n = len(X)
    if isinstance(X, pd.DataFrame):
        col = pd.DataFrame({"_w": np.full(n, value)}, index=X.index)
        return pd.concat([col, X], axis=1)
    if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
        col = pl.Series("_w", np.full(n, value))
        return X.with_columns(col).select(["_w"] + X.columns)
    # numpy
    return np.hstack((np.full((n, 1), value), X))


def concat_treatment_col(w, X):
    """Prepend an array-valued treatment column *w* to feature matrix X.

    Unlike :func:`prepend_column`, this takes an existing 1-D array rather
    than a scalar. Used by S-learner ``fit()`` to build the augmented matrix.

    Args:
        w: 1-D numpy array of treatment indicators (0/1).
        X: numpy array, pd.DataFrame, or pl.DataFrame.

    Returns:
        Augmented feature matrix with w prepended, same type as X.
    """
    if isinstance(X, pd.DataFrame):
        col = pd.DataFrame({"_w": w}, index=X.index)
        return pd.concat(
            [col, X.reset_index(drop=True) if not X.index.equals(col.index) else X],
            axis=1,
        )
    if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
        col = pl.Series("_w", np.asarray(w))
        return X.with_columns(col).select(["_w"] + X.columns)
    # numpy
    return np.hstack((np.asarray(w).reshape(-1, 1), X))


def to_numpy(obj):
    """Convert a pandas or polars object to a NumPy array.

    Use this *only* at boundaries where a third-party library strictly
    requires numpy (rare — sklearn >= 1.6 and XGBoost >= 3.1 both accept
    DataFrames natively).

    Args:
        obj: numpy array, pd.DataFrame, pd.Series, pl.DataFrame,
             pl.LazyFrame, or pl.Series.  ``None`` is passed through.

    Returns:
        numpy.ndarray or None.
    """
    if obj is None:
        return None
    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series)):
        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()
        arr = obj.to_numpy()
        if isinstance(obj, pl.DataFrame) and arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr
    if hasattr(obj, "to_numpy"):
        return obj.to_numpy()
    return np.asarray(obj)


# ---------------------------------------------------------------------------
# Legacy helper — kept for backward compatibility with external callers.
# Internal learner code now uses filter_mask / filter_index / to_numpy.
# ---------------------------------------------------------------------------


def convert_pd_to_np(*args):
    """Convert pandas or polars objects to NumPy arrays.

    .. deprecated::
        Internal learner code no longer calls this function.  It is kept
        solely for backward compatibility with user code that imports it
        directly.  New code should use :func:`to_numpy` instead.
    """
    output = [to_numpy(obj) for obj in args]
    return output if len(output) > 1 else output[0]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_treatment_vector(treatment, control_name=None):
    """Assert that *treatment* has at least two unique levels."""
    # Normalise to numpy for np.unique
    t = to_numpy(treatment) if not isinstance(treatment, np.ndarray) else treatment
    n_unique_treatments = np.unique(t).shape[0]
    assert n_unique_treatments > 1, "Treatment vector must have at least two levels."
    if control_name is not None:
        assert (
            control_name in t
        ), "Control group level {} not found in treatment vector.".format(control_name)


def check_p_conditions(p, t_groups):
    eps = np.finfo(float).eps

    _allowed = [np.ndarray, pd.Series]
    if _POLARS_AVAILABLE:
        _allowed.append(pl.Series)
    _allowed_tuple = tuple(_allowed)

    assert isinstance(
        p, (*_allowed_tuple, dict)
    ), "p must be an np.ndarray, pd.Series, pl.Series (if polars is installed), or dict type"

    if isinstance(p, _allowed_tuple):
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


# ---------------------------------------------------------------------------
# XGBoost objective helpers (unchanged)
# ---------------------------------------------------------------------------


def clean_xgboost_objective(objective):
    """Translate objective to be compatible with the loaded xgboost version."""
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
    """Return version-compatible (objective, eval_metric) tuple."""

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
    """Calculate the variance of array x with sample_weight."""
    average = np.average(x, weights=sample_weight)
    variance = np.average((x - average) ** 2, weights=sample_weight)
    return variance
