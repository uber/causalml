"""Kernel-backed uplift random forest classifier.

Hosts the private ``_KernelUpliftRandomForestClassifier`` (bags
``_KernelUpliftTreeClassifier`` on scikit-learn's ``ForestRegressor``
scaffolding, mirroring ``CausalRandomForestRegressor``) and the public
:class:`UpliftRandomForestClassifier`, a thin backward-compatible subclass of it
(issue #955 switchover).

Like the legacy forest it bootstraps by resampling rows (so a tree can miss a
treatment group entirely -- the #569 sparse-group case handled by
``_align_tree_predict``), keeps the ``joblib_prefer`` knob, and predicts the
per-treatment delta columns (``full_output`` returns the full frame). Early
stopping on a validation set -- a legacy tree feature -- is not supported here
because the kernel tree does not implement it.
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import float32 as DTYPE
from sklearn.ensemble._forest import ForestRegressor, MAX_INT
from sklearn.utils import check_array, check_random_state

from causalml.inference.serialization import SerializableLearner

from .uplifttree import _KernelUpliftTreeClassifier


def _align_tree_predict(tree, X, forest_classes, class_to_forest_idx):
    """Predict per-group P(Y=1|T=g) with one tree, aligned to the forest classes.

    A row-subsampled bootstrap can exclude whole treatment groups, so a tree's
    ``classes_`` may be a strict subset of the forest's. This scatters the
    tree's per-group columns into the forest-wide ordering (zero-filling absent
    groups) before the forest averages -- the #569 sparse-group case. A leaf
    with no samples of an otherwise-present group yields a NaN rate under
    ``min_samples_treatment=0``; those are zeroed so they do not poison the
    ensemble average (the legacy tree instead backed such leaves up to the
    parent rate).
    """
    raw = np.nan_to_num(tree.predict_proba_by_group(X=X), nan=0.0)
    if len(tree.classes_) == len(forest_classes):
        return raw
    aligned = np.zeros((raw.shape[0], len(forest_classes)), dtype=raw.dtype)
    for tree_idx, cls in enumerate(tree.classes_):
        forest_idx = class_to_forest_idx.get(cls)
        if forest_idx is not None:
            aligned[:, forest_idx] = raw[:, tree_idx]
    return aligned


def _parallel_build_tree(tree, X, treatment, y, control_name, sample_weight):
    """Fit one tree on a row-resampled bootstrap of ``(X, treatment, y)``.

    Mirrors the legacy forest's bootstrap: draw ``len(X)`` rows with replacement
    from the tree's own ``random_state`` (so a bootstrap can miss treatment
    groups), then fit the kernel tree on that subset. A bootstrap that lands on
    the control group alone (or drops the control group) carries no
    treatment-effect signal and cannot grow an uplift tree, so it is dropped by
    returning ``None`` -- only possible under extreme group imbalance (the #569
    case). The legacy forest instead kept such trees as no-ops.
    """
    rng = check_random_state(tree.random_state)
    idx = rng.choice(len(X), len(X))
    t_sub = treatment[idx]
    groups = set(t_sub)
    if control_name not in groups or len(groups) < 2:
        return None
    sw = None if sample_weight is None else sample_weight[idx]
    tree.fit(X[idx], t_sub, y[idx], sample_weight=sw, check_input=True)
    return tree


class _KernelUpliftRandomForestClassifier(SerializableLearner, ForestRegressor):
    """A random forest of kernel-backed uplift trees.

    Bags ``_KernelUpliftTreeClassifier`` on sklearn's ``ForestRegressor``
    scaffolding. ``predict`` returns the per-treatment uplift deltas
    ``P(Y=1|T=t) - P(Y=1|control)``; ``predict(..., full_output=True)`` returns
    the full frame (per-group probabilities, recommended treatment, deltas,
    ``max_delta``).

    Inherits :class:`~causalml.inference.serialization.SerializableLearner` for
    ``save`` / ``load`` and is a scikit-learn ``BaseEstimator`` via
    ``ForestRegressor`` (``get_params`` / ``clone`` round-trip). Full
    ``check_estimator`` does not apply -- ``fit`` takes ``(X, treatment, y)``.
    """

    def __init__(
        self,
        control_name: Union[int, str] = "control",
        n_estimators: int = 10,
        *,
        criterion: str = "KL",
        max_depth: int = 5,
        min_samples_leaf: int = 100,
        min_samples_split: Union[int, float] = 2,
        min_samples_treatment: int = 10,
        n_reg: int = 10,
        normalization: bool = True,
        honesty: bool = False,
        estimation_sample_size: float = 0.5,
        max_features: Union[int, float, str, None] = "sqrt",
        min_weight_fraction_leaf: float = 0.0,
        n_jobs: int = -1,
        random_state: int = None,
        joblib_prefer: str = "threads",
    ):
        estimator = _KernelUpliftTreeClassifier(
            criterion=criterion,
            control_name=control_name,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_samples_treatment=min_samples_treatment,
            n_reg=n_reg,
            normalization=normalization,
            honesty=honesty,
            estimation_sample_size=estimation_sample_size,
            max_features=max_features,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
        )
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "control_name",
                "max_depth",
                "min_samples_leaf",
                "min_samples_split",
                "min_samples_treatment",
                "n_reg",
                "normalization",
                "honesty",
                "estimation_sample_size",
                "max_features",
                "min_weight_fraction_leaf",
                "random_state",
            ),
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.criterion = criterion
        self.control_name = control_name
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.normalization = normalization
        self.honesty = honesty
        self.estimation_sample_size = estimation_sample_size
        self.max_features = max_features
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.joblib_prefer = joblib_prefer

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        """Fit the forest.

        Args:
            X (np.ndarray): feature matrix
            treatment (np.ndarray): treatment vector, includes the control group
            y (np.ndarray): binary outcome vector
            sample_weight (np.ndarray): optional sample weights
        Returns:
            self
        """
        # NaNs are handled natively by the kernel trees (see
        # ``_KernelUpliftTreeClassifier``); keep them through forest validation.
        X = check_array(
            X, accept_sparse=False, dtype=DTYPE, ensure_all_finite="allow-nan"
        )
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]

        # Forest-wide class ordering; control is reserved for column 0.
        treatment_groups = sorted(g for g in set(treatment) if g != self.control_name)
        self.classes_ = [self.control_name] + treatment_groups
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_

        self._validate_estimator()
        rng = check_random_state(self.random_state)

        # Build the trees, seeding each from the parent RNG in the same order as
        # the legacy forest so the per-tree bootstrap draws line up.
        trees = [
            self._make_estimator(append=False, random_state=None)
            for _ in range(self.n_estimators)
        ]
        for tree in trees:
            tree.random_state = rng.randint(MAX_INT)

        trees = Parallel(n_jobs=self.n_jobs, prefer=self.joblib_prefer)(
            delayed(_parallel_build_tree)(
                tree, X, treatment, y, self.control_name, sample_weight
            )
            for tree in trees
        )
        self.estimators_ = [tree for tree in trees if tree is not None]
        return self

    def predict(self, X: np.ndarray, full_output: bool = False):
        """Predict per-treatment uplift.

        Args:
            X (np.ndarray): feature matrix
            full_output (bool): if True return the full frame (per-group
                probabilities, recommended treatment, deltas, ``max_delta``);
                otherwise return only the delta columns as an array.
        Returns:
            np.ndarray of shape (n_samples, n_treatments), or a pandas.DataFrame
            when ``full_output`` is True.
        """
        class_to_forest_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        preds = Parallel(n_jobs=self.n_jobs, prefer=self.joblib_prefer)(
            delayed(_align_tree_predict)(tree, X, self.classes_, class_to_forest_idx)
            for tree in self.estimators_
        )
        y_pred_ensemble = sum(preds) / len(self.estimators_)

        df_res = pd.DataFrame(y_pred_ensemble, columns=self.classes_)
        df_res["recommended_treatment"] = y_pred_ensemble.argmax(axis=1)

        delta_cols = [f"delta_{group}" for group in self.classes_[1:]]
        for group in self.classes_[1:]:
            df_res[f"delta_{group}"] = df_res[group] - df_res[self.control_name]
        df_res["max_delta"] = df_res[delta_cols].max(axis=1)

        if full_output:
            return df_res
        return df_res[delta_cols].values


class UpliftRandomForestClassifier(_KernelUpliftRandomForestClassifier):
    """Uplift random forest classifier.

    Kernel-backed drop-in for the historical Cython
    ``UpliftRandomForestClassifier`` (issue #955 switchover). The legacy
    constructor names and defaults are preserved: ``evaluationFunction`` selects
    the split criterion, ``max_features`` defaults to ``10`` (clamped to the
    feature count, as the legacy forest did), and the fitted trees are exposed as
    ``uplift_forest``. ``predict`` returns the per-treatment uplift deltas
    (``full_output=True`` returns the full frame). Bags kernel-backed uplift
    trees on scikit-learn's ``ForestRegressor`` (see
    :class:`_KernelUpliftRandomForestClassifier`).

    ``early_stopping_eval_diff_scale`` and ``fit``'s ``X_val`` / ``treatment_val``
    / ``y_val`` are accepted for backward compatibility but ignored: validation-set
    early stopping is not implemented on the kernel trees.
    """

    def __init__(
        self,
        control_name,
        n_estimators=10,
        max_features=10,
        random_state=None,
        max_depth=5,
        min_samples_leaf=100,
        min_samples_treatment=10,
        n_reg=10,
        early_stopping_eval_diff_scale=1,
        evaluationFunction="KL",
        normalization=True,
        honesty=False,
        estimation_sample_size=0.5,
        n_jobs=-1,
        joblib_prefer: str = "threads",
    ):
        # Retained verbatim for the sklearn get_params / clone contract on the
        # legacy signature (mapped to the kernel ``criterion`` below).
        self.evaluationFunction = evaluationFunction
        # Legacy validation-set early stopping is not implemented on the kernel
        # trees; kept for backward-compatible construction only.
        self.early_stopping_eval_diff_scale = early_stopping_eval_diff_scale
        super().__init__(
            control_name=control_name,
            n_estimators=n_estimators,
            criterion=evaluationFunction,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_treatment=min_samples_treatment,
            n_reg=n_reg,
            normalization=normalization,
            honesty=honesty,
            estimation_sample_size=estimation_sample_size,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            joblib_prefer=joblib_prefer,
        )

    @property
    def uplift_forest(self):
        """The fitted trees (legacy attribute name; aliases ``estimators_``)."""
        return self.estimators_

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        treatment_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ):
        """Fit the forest. ``X_val`` / ``treatment_val`` / ``y_val`` (legacy early
        stopping) are accepted for backward compatibility but ignored."""
        if X_val is not None or treatment_val is not None or y_val is not None:
            warnings.warn(
                "Validation-set early stopping (X_val / treatment_val / y_val, "
                "early_stopping_eval_diff_scale) is not supported by the "
                "kernel-backed UpliftRandomForestClassifier and is ignored.",
                UserWarning,
            )
        # The legacy forest clamps max_features to the feature count; the kernel
        # tree validates 0 < max_features <= n_features strictly. Clamp for the
        # tree build, then restore so get_params / clone report the original.
        n_features = np.asarray(X).shape[1]
        orig_max_features = self.max_features
        if (
            isinstance(self.max_features, (int, np.integer))
            and not isinstance(self.max_features, bool)
            and self.max_features > n_features
        ):
            self.max_features = int(n_features)
        try:
            super().fit(X, treatment, y, sample_weight=sample_weight)
        finally:
            self.max_features = orig_max_features
        return self
