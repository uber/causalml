"""Experimental kernel-backed uplift random forest classifier.

Not part of the public API -- this bags the experimental
``_KernelUpliftTreeClassifier`` on scikit-learn's ``ForestRegressor``
scaffolding (mirroring ``CausalRandomForestRegressor``) to prove parity with
the legacy joblib-bagging ``UpliftRandomForestClassifier`` before the public
class is switched over.

Like the legacy forest it bootstraps by resampling rows (so a tree can miss a
treatment group entirely -- the #569 sparse-group case handled by
``_align_tree_predict``), keeps the ``joblib_prefer`` knob, and predicts the
per-treatment delta columns (``full_output`` returns the full frame). Early
stopping on a validation set -- a legacy tree feature -- is not supported here
because the kernel tree does not implement it.
"""

from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import float32 as DTYPE
from sklearn.ensemble._forest import ForestRegressor, MAX_INT
from sklearn.utils import check_array, check_random_state

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


class _KernelUpliftRandomForestClassifier(ForestRegressor):
    """A random forest of kernel-backed uplift trees.

    Bags ``_KernelUpliftTreeClassifier`` on sklearn's ``ForestRegressor``
    scaffolding. ``predict`` returns the per-treatment uplift deltas
    ``P(Y=1|T=t) - P(Y=1|control)``; ``predict(..., full_output=True)`` returns
    the full frame (per-group probabilities, recommended treatment, deltas,
    ``max_delta``).
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
        X = check_array(X, accept_sparse=False, dtype=DTYPE)
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
