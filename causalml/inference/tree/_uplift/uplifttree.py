"""Experimental kernel-backed uplift tree classifier.

Not part of the public API -- this exists to prove numerical parity of the
kernel-backed KL / ED / Chi / CTS / DDP / IT / CIT / IDDP criteria against the
legacy ``UpliftTreeClassifier`` before the public classes are switched over. It
supports the Rzepakowski ``n_reg`` / ``min_samples_treatment`` regularization,
``normalization``, and the honest approach (held-out leaf re-estimation, Athey &
Imbens 2016); the two-class criteria (DDP/IT/CIT/IDDP) reject multi-treatment
input and IDDP forces honesty on. It also exposes a legacy-shaped
``fitted_uplift_tree`` so ``plot.py`` renders it unchanged (issue #953). Pruning
and the forest are handled in other issues of the epic.
"""

from typing import Union

import numpy as np
from numpy import float32 as DTYPE
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from causalml.inference.meta.utils import check_treatment_vector

from ._tree import BaseUpliftDecisionTree


class _UpliftTreeNode:
    """A minimal ``DecisionTree``-shaped node for ``plot.py``.

    ``plot.py``'s ``uplift_tree_string`` / ``uplift_tree_plot`` duck-type the
    legacy pure-Python ``DecisionTree`` node. This exposes exactly the fields
    they read (``classes_``, ``col``, ``value``, ``trueBranch``, ``falseBranch``,
    ``results``, ``summary``) so the plot helpers work unchanged against the
    kernel tree; it does not depend on the legacy node (which is removed at the
    epic switchover).
    """

    def __init__(
        self,
        classes_,
        col=-1,
        value=None,
        trueBranch=None,
        falseBranch=None,
        results=None,
        summary=None,
    ):
        self.classes_ = classes_
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # per-group P(Y=1|T) on a leaf, None on a split
        self.summary = summary


class _KernelUpliftTreeClassifier(BaseUpliftDecisionTree):
    """A single uplift tree grown on the shared ``_tree`` Cython kernel."""

    def __init__(
        self,
        *,
        criterion: str = "KL",
        control_name: Union[int, str] = "control",
        max_depth: int = 3,
        min_samples_leaf: int = 100,
        min_samples_split: Union[int, float] = 2,
        min_samples_treatment: int = 10,
        n_reg: int = 100,
        normalization: bool = True,
        honesty: bool = False,
        estimation_sample_size: float = 0.5,
        max_features: Union[int, float, str, None] = None,
        min_weight_fraction_leaf: float = 0.0,
        random_state: int = None,
    ):
        self.control_name = control_name
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.normalization = normalization
        # IDDP requires the honest approach (legacy uplift.pyx ~468-469).
        if criterion == "IDDP" and not honesty:
            honesty = True
        self.honesty = honesty
        self.estimation_sample_size = estimation_sample_size
        super().__init__(
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=None,
            random_state=random_state,
            min_impurity_decrease=0.0,
            ccp_alpha=0.0,
        )

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: Union[np.ndarray, None] = None,
        check_input: bool = True,
    ):
        """Fit the uplift tree.

        Args:
            X (np.ndarray): feature matrix
            treatment (np.ndarray): treatment vector, includes the control group
            y (np.ndarray): binary outcome vector (coerced to {0, 1})
            sample_weight (np.ndarray): optional sample weights
            check_input (bool): default True
        Returns:
            self
        """
        X_enc, y_2dim = self._prepare_data(X=X, treatment=treatment, y=y)

        if not self.honesty:
            super().fit(
                X=X_enc, y=y_2dim, sample_weight=sample_weight, check_input=check_input
            )
            self._node_group_counts = self._compute_node_group_counts(X_enc, y_2dim)
            return self

        # Honest approach (Athey & Imbens 2016): grow the tree on one split and
        # re-estimate the leaf probabilities on a held-out estimation split. The
        # split mirrors the legacy tree exactly -- stratified on (treatment, y),
        # same test_size / shuffle / random_state -- so the two trees partition
        # the data identically.
        treatment = np.asarray(treatment)
        treatment_idx = np.fromiter(
            (self._group2index[t] for t in treatment), dtype=int, count=len(treatment)
        )
        y_bin = (np.asarray(y).ravel() > 0).astype(int)
        stratify = np.stack([treatment_idx, y_bin], axis=1)

        arrays = [X_enc, y_2dim]
        if sample_weight is not None:
            arrays.append(sample_weight)
        try:
            split = train_test_split(
                *arrays,
                stratify=stratify,
                test_size=self.estimation_sample_size,
                shuffle=True,
                random_state=self.random_state,
            )
        except ValueError:
            split = train_test_split(
                *arrays,
                test_size=self.estimation_sample_size,
                shuffle=True,
                random_state=self.random_state,
            )
        X_tr, X_est, y_tr, y_est = split[0], split[1], split[2], split[3]
        sw_tr = split[4] if sample_weight is not None else None

        super().fit(X=X_tr, y=y_tr, sample_weight=sw_tr, check_input=check_input)
        self._honest_reestimate(X_est, y_est)
        # Per-group counts for plotting come from the structure (train) split.
        self._node_group_counts = self._compute_node_group_counts(X_tr, y_tr)
        return self

    def _honest_reestimate(self, X_est: np.ndarray, y_est: np.ndarray) -> None:
        """Overwrite each leaf's per-group P(Y=1|T=g) on the estimation split.

        Routes the estimation rows through the grown tree and, for every leaf,
        recomputes each group's rate as the raw ``n_pos / n`` over the estimation
        rows reaching that leaf (0.0 when a group is absent) -- matching legacy
        ``fillTree`` / ``uplift_classification_results``. ``tree_.value`` shares
        memory with the tree, so the assignment mutates the leaf estimates the
        predictions read.
        """
        X_est = np.ascontiguousarray(X_est, dtype=DTYPE)
        leaf_ids = self.tree_.apply(X_est)
        value = self.tree_.value  # (node_count, n_groups, 1), writable view
        n_nodes = self.tree_.node_count
        is_leaf = self.tree_.children_left == -1

        for g in range(self.n_outputs_):
            col = y_est[:, g]
            valid = ~np.isnan(col)
            leaves = leaf_ids[valid]
            n = np.bincount(leaves, minlength=n_nodes).astype(np.float64)
            n_pos = np.bincount(leaves, weights=col[valid], minlength=n_nodes)
            p = np.divide(n_pos, n, out=np.zeros_like(n), where=n > 0)
            value[is_leaf, g, 0] = p[is_leaf]

    def _compute_node_group_counts(self, X_enc: np.ndarray, y_2dim: np.ndarray):
        """Per-node, per-group sample counts N(T=g) reaching each node.

        The kernel ``Tree`` stores only total ``n_node_samples`` per node, but the
        plot's ``group_size`` line and the uplift-score p-value need per-group
        counts. Route the fit rows through the tree once (``decision_path``) and
        tally each group by its non-NaN column in ``y_2dim`` -- the same routing
        the honest re-estimation uses. Returns an ``(n_nodes, n_groups)`` array.
        """
        dpath = self.tree_.decision_path(np.ascontiguousarray(X_enc, dtype=DTYPE))
        counts = np.zeros((self.tree_.node_count, self.n_outputs_), dtype=np.int64)
        for g in range(self.n_outputs_):
            mask = ~np.isnan(y_2dim[:, g])
            counts[:, g] = np.asarray(dpath[mask].sum(axis=0)).ravel()
        return counts

    @staticmethod
    def _uplift_score(p: np.ndarray, n: np.ndarray) -> tuple:
        """Legacy node uplift score ``[maxDiff, p_value]`` from per-group P and N.

        ``maxDiff`` is the largest ``P(Y=1|T=t) - P(Y=1|control)`` over treatments;
        ``p_value`` is the legacy two-proportion z-test between control and the
        best (or, if none beat control, least-bad) treatment (uplift.pyx ~2109).
        """
        p_c, n_c = p[0], n[0]
        max_diff, best, subopt = -1.0, 0, 0
        for i in range(1, len(p)):
            diff = p[i] - p_c
            if diff >= max_diff:
                max_diff, subopt = diff, i
                if diff > 0:
                    best = i
        idx = best if max_diff > 0 else subopt
        p_t, n_t = p[idx], n[idx]
        if n_t > 0 and n_c > 0:
            variance = p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c
            p_value = (
                (1.0 - norm.cdf(abs(p_c - p_t) / np.sqrt(variance))) * 2
                if variance > 0
                else 1.0
            )
        else:
            p_value = 1.0
        return max_diff, p_value

    @property
    def fitted_uplift_tree(self) -> "_UpliftTreeNode":
        """The fitted tree as a legacy-``DecisionTree``-shaped node, for plotting.

        Mirrors the legacy ``UpliftTreeClassifier.fitted_uplift_tree`` attribute
        so ``uplift_tree_plot`` / ``uplift_tree_string`` render the kernel tree
        unchanged. Built lazily from the kernel ``Tree`` node arrays and the
        per-group counts recorded at fit.
        """
        check_is_fitted(self, "tree_")
        return self._build_plot_node(0)

    def _build_plot_node(self, node_id: int) -> "_UpliftTreeNode":
        """Recursively build the plot node for ``node_id`` and its subtree."""
        tree = self.tree_
        p = tree.value[node_id, :, 0]  # per-group P(Y=1|T=g)
        n = self._node_group_counts[node_id]  # per-group N(T=g)
        max_diff, p_value = self._uplift_score(p, n)
        summary = {
            "impurity": "%.3f" % tree.impurity[node_id],
            "samples": "%d" % tree.n_node_samples[node_id],
            "group_size": "".join(
                " %s: %d" % (cls, cnt) for cls, cnt in zip(self.classes_, n)
            ),
            "upliftScore": [round(float(max_diff), 4), round(float(p_value), 4)],
            "matchScore": round(float(max_diff), 4),
        }

        left, right = tree.children_left[node_id], tree.children_right[node_id]
        if left == -1:  # leaf
            return _UpliftTreeNode(
                classes_=self.classes_, results=list(p), summary=summary
            )
        # Legacy renders "col >= value? yes -> trueBranch"; the kernel sends
        # X[col] > threshold to the right child, so right is the "yes" branch.
        return _UpliftTreeNode(
            classes_=self.classes_,
            col=int(tree.feature[node_id]),
            value=float(tree.threshold[node_id]),
            trueBranch=self._build_plot_node(right),
            falseBranch=self._build_plot_node(left),
            summary=summary,
        )

    def predict_proba_by_group(
        self, X: np.ndarray, check_input: bool = True
    ) -> np.ndarray:
        """Per-group leaf probabilities P(Y=1 | T=g).

        Returns:
            np.ndarray, shape (n_samples, n_groups); column 0 is the control
            group, columns 1..k the treatment groups in sorted order.
        """
        return BaseUpliftDecisionTree.predict(self, X, check_input=check_input)

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Per-treatment individual treatment effect P(Y=1|T=t) - P(Y=1|control).

        Returns:
            np.ndarray, shape (n_samples, n_treatments).
        """
        proba = self.predict_proba_by_group(X, check_input=check_input)
        return proba[:, 1:] - proba[:, [0]]

    def _prepare_data(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Encode (X, treatment, y) as (X, group-matrix y).

        The outcome is coerced to binary and reshaped to a
        ``(n_samples, n_groups)`` NaN-masked matrix with the control group in
        column 0 -- the same encoding the causal trees use.
        """
        if y.shape[0] != treatment.shape[0]:
            raise ValueError(
                f"The number of `treatment` and `y` rows are not equal: "
                f"{y.shape[0]} {treatment.shape[0]}"
            )
        check_treatment_vector(treatment, self.control_name)
        self.unique_groups = list(set(treatment))
        self.unique_treatments = sorted(
            [x for x in self.unique_groups if x != self.control_name]
        )
        self._group2index = {
            self.control_name: 0,
            **{treatment: i + 1 for i, treatment in enumerate(self.unique_treatments)},
        }
        self.classes_ = [self.control_name] + self.unique_treatments

        X = check_array(X, dtype=DTYPE, accept_sparse="csc")
        y = check_array(y, ensure_2d=False, dtype=None)
        y = (y > 0).astype(np.float64)
        self.n_samples, self.n_features = X.shape

        y_2dim = np.zeros((self.n_samples, len(self.unique_treatments) + 1))
        for group, group_index in self._group2index.items():
            y_2dim[:, group_index] = np.where(treatment == group, y, np.nan)

        return X, y_2dim
