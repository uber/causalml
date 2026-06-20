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
#
# Design contract (per maintainer review):
#   - X (feature matrices) stay native end-to-end: numpy, pandas, or polars.
#     pl.LazyFrame is collected ONCE at the top of each public method into a
#     pl.DataFrame via `collect_if_lazy`. After that point, helpers only need
#     to handle pl.DataFrame / pl.Series.
#   - treatment / y / p / sample_weight (1-D vectors used for masking,
#     np.unique, .astype, etc.) are normalized to numpy via `to_numpy` at the
#     entry of each public method.
#   - Never call to_numpy(X) merely to read a row count: use `n_rows(X)`,
#     which works for numpy, pandas, and polars.
# ---------------------------------------------------------------------------


def collect_if_lazy(X):
    """Collect a polars LazyFrame into a DataFrame; pass through otherwise.

    Call this once at the top of each public method on the feature matrix X
    so that downstream helpers (filter_mask, filter_index, prepend_column,
    concat_treatment_col, n_rows) never need to handle pl.LazyFrame.

    Args:
        X: numpy array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame.

    Returns:
        X unchanged, unless it was a pl.LazyFrame (then its .collect()).
    """
    if _POLARS_AVAILABLE and isinstance(X, pl.LazyFrame):
        return X.collect()
    return X


def n_rows(X):
    """Return the number of rows of X without converting to numpy.

    Works for numpy arrays, pandas DataFrames/Series, and polars
    DataFrames/Series (and LazyFrames, which are collected first).

    Args:
        X: numpy array, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series,
           or pl.LazyFrame.

    Returns:
        int: number of rows.
    """
    X = collect_if_lazy(X)
    if hasattr(X, "shape"):
        return X.shape[0]
    return len(X)


def filter_mask(obj, mask):
    """Filter rows by a boolean mask.

    Works natively for numpy arrays, pandas DataFrames/Series, and
    polars DataFrames/Series/LazyFrames without materialising unnecessary
    copies.

    Args:
        obj: numpy array, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series,
             or pl.LazyFrame.
        mask: boolean array or Series of the same length as obj.

    Returns:
        Filtered object of the same type as input (a pl.LazyFrame input is
        returned as a pl.DataFrame, since filtering requires collection).
    """
    if obj is None:
        return None

    obj = collect_if_lazy(obj)

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(mask, pd.Series):
            return obj.loc[mask]
        return obj.loc[np.asarray(mask, dtype=bool)]

    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.Series)):
        if isinstance(mask, pl.Series):
            return obj.filter(mask)
        return obj.filter(pl.Series(np.asarray(mask, dtype=bool)))

    # numpy / anything else
    return obj[np.asarray(mask, dtype=bool)]


def filter_index(obj, indices):
    """Filter rows by integer positional indices.

    Used for KFold partition slicing (e.g. in DR-learner) and bootstrap
    resampling.

    Args:
        obj: numpy array, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series,
             or pl.LazyFrame.
        indices: integer array of row positions.

    Returns:
        Filtered object of the same type as input (a pl.LazyFrame input is
        returned as a pl.DataFrame).
    """
    if obj is None:
        return None

    obj = collect_if_lazy(obj)

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[indices]

    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.Series)):
        return obj[indices]

    return obj[indices]  # numpy


def prepend_column(value, X):
    """Prepend a constant-value column to a feature matrix X.

    Used by the S-learner to prepend the treatment indicator before
    calling predict() for the control (value=0.0) and treatment (value=1.0)
    counterfactuals.

    Args:
        value (float): scalar value to fill the new column with (0.0 or 1.0).
        X: numpy array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame.

    Returns:
        Feature matrix with the new column prepended, same type as X (a
        pl.LazyFrame input is returned as a pl.DataFrame).
    """
    X = collect_if_lazy(X)
    n = n_rows(X)

    if isinstance(X, pd.DataFrame):
        col_name = X.columns[0].__class__(0) if len(X.columns) > 0 else "_w"
        col = pd.DataFrame({col_name: np.full(n, value)}, index=X.index)
        return pd.concat([col, X], axis=1)

    if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
        col = pl.Series("_w", np.full(n, value))
        return X.with_columns(col).select(["_w"] + X.columns)

    # numpy
    return np.hstack((np.full((n, 1), value), X))


def concat_treatment_col(w, X):
    """Prepend an array-valued treatment column *w* to feature matrix X.

    Unlike :func:`prepend_column`, this takes an existing 1-D array rather
    than a scalar. Used by S-learner ``fit()`` to build the augmented
    matrix [w | X].

    Args:
        w: 1-D numpy array of treatment indicators (0/1).
        X: numpy array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame.

    Returns:
        Augmented feature matrix with w prepended, same type as X (a
        pl.LazyFrame input is returned as a pl.DataFrame).
    """
    X = collect_if_lazy(X)

    if isinstance(X, pd.DataFrame):
        col_name = X.columns[0].__class__(0) if len(X.columns) > 0 else "_w"
        col = pd.DataFrame({col_name: np.asarray(w)}, index=X.index)
        return pd.concat([col, X], axis=1)

    if _POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
        col = pl.Series("_w", np.asarray(w))
        return X.with_columns(col).select(["_w"] + X.columns)

    # numpy
    return np.hstack((np.asarray(w).reshape(-1, 1), X))


def to_numpy(obj):
    """Convert a pandas or polars 1-D vector (or feature matrix) to NumPy.

    Per the native-X contract, this should be called on ``treatment``,
    ``y``, ``p``, and ``sample_weight`` at the entry of each public method.
    It should NOT be called on the feature matrix X except at the small
    number of boundaries that strictly require numpy (documented inline
    where used).

    Args:
        obj: numpy array, pd.DataFrame, pd.Series, pl.DataFrame,
             pl.LazyFrame, or pl.Series.  ``None`` is passed through.

    Returns:
        numpy.ndarray or None.
    """
    if obj is None:
        return None

    obj = collect_if_lazy(obj)

    if _POLARS_AVAILABLE and isinstance(obj, (pl.DataFrame, pl.Series)):
        arr = obj.to_numpy()
        if isinstance(obj, pl.DataFrame) and arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr

    if hasattr(obj, "to_numpy"):
        return obj.to_numpy()

    return np.asarray(obj)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_treatment_vector(treatment, control_name=None):
    """Assert that *treatment* has at least two unique levels.

    Args:
        treatment (np.array, pd.Series, or pl.Series): a treatment vector.
        control_name (str or int, optional): name of the control group.
    """
    t = to_numpy(treatment)
    n_unique_treatments = np.unique(t).shape[0]
    assert n_unique_treatments > 1, "Treatment vector must have at least two levels."
    if control_name is not None:
        assert (
            control_name in t
        ), "Control group level {} not found in treatment vector.".format(control_name)


def check_p_conditions(p, t_groups):
    """Validate that propensity scores p lie strictly within (0, 1).

    Args:
        p (np.ndarray, pd.Series, pl.Series, or dict): propensity scores.
        t_groups (np.ndarray): unique non-control treatment group labels.
    """
    eps = np.finfo(float).eps

    _allowed = [np.ndarray, pd.Series]
    if _POLARS_AVAILABLE:
        _allowed.append(pl.Series)
    _allowed_tuple = tuple(_allowed)

    assert isinstance(
        p, (*_allowed_tuple, dict)
    ), "p must be an np.ndarray, pd.Series, pl.Series (if polars is installed), or dict type"

    if isinstance(p, _allowed_tuple):
        p_np = to_numpy(p)
        assert (
            t_groups.shape[0] == 1
        ), "If p is passed as an array/Series, there must be only 1 unique non-control group in the treatment vector."
        assert (0 + eps < p_np).all() and (
            p_np < 1 - eps
        ).all(), "The values of p should lie within the (0, 1) interval."

    if isinstance(p, dict):
        for t_name in t_groups:
            p_np = to_numpy(p[t_name])
            assert (0 + eps < p_np).all() and (
                p_np < 1 - eps
            ).all(), "The values of p should lie within the (0, 1) interval."


def check_explain_conditions(method, models, X=None, treatment=None, y=None):
    """Validate inputs for explain_validation-style feature-importance methods.

    Args:
        method (str): one of "gini", "permutation", "shapley".
        models (list): models that need a ``.feature_importances_`` attribute
            for "gini"/"shapley".
        X, treatment, y: required (non-None) for "permutation"/"shapley".
    """
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


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------


def convert_pd_to_np(*args):
    """Deprecated alias for :func:`to_numpy`, kept for backward compatibility
    with modules (e.g. explainer.py) that have not yet migrated to the
    native-DataFrame contract.

    .. deprecated::
        Use :func:`to_numpy` for 1-D vectors. Do not use on feature matrices
        X under the native-X contract.
    """
    output = [to_numpy(obj) for obj in args]
    return output if len(output) > 1 else output[0]
