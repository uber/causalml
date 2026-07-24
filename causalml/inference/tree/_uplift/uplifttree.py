"""Experimental kernel-backed uplift tree classifier.

Not part of the public API -- this exists to prove numerical parity of the
kernel-backed KL / ED / Chi / CTS / DDP / IT / CIT / IDDP criteria against the
legacy ``UpliftTreeClassifier`` before the public classes are switched over. It
supports the Rzepakowski ``n_reg`` / ``min_samples_treatment`` regularization,
``normalization``, the honest approach (held-out leaf re-estimation, Athey &
Imbens 2016), and validation-based pruning; the two-class criteria
(DDP/IT/CIT/IDDP) reject multi-treatment input and IDDP forces honesty on. The
forest is handled in a later issue of the epic.
"""

from typing import Union

import numpy as np
from numpy import float32 as DTYPE
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array

from causalml.inference.meta.utils import check_treatment_vector

from ._tree import BaseUpliftDecisionTree
from .._tree._tree import Tree, build_pruned_tree_from_mask


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

    def prune(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        minGain: float = 0.0001,
        rule: str = "maxAbsDiff",
    ):
        """Validation-based bottom-up merge pruning.

        Collapses an internal node whose two children are leaves into a leaf
        when the split does not improve the treatment-effect estimate on the
        held-out validation set ``(X, treatment, y)``, cascading upward until no
        further merge helps, then rebuilds a compact tree.

        This reimplements the *intent* of the legacy
        ``UpliftTreeClassifier.prune`` on the kernel tree. (The legacy recursion
        never descended past the root -- its guard only recurses when a child is
        missing -- so legacy pruning is a no-op; there is no legacy behavior to
        match bit-for-bit here.)

        Args:
            X (np.ndarray): validation feature matrix.
            treatment (np.ndarray): validation treatment vector (incl. control).
            y (np.ndarray): validation binary outcome vector.
            minGain (float): a split is kept only if it raises the summed child
                uplift over the parent by more than this.
            rule (str): ``"maxAbsDiff"`` (max-abs-diff treatment, signed,
                child-size weighted) or ``"bestUplift"`` (best positive-uplift
                treatment, total sample-fraction weighted).
        Returns:
            self
        """
        if rule not in ("maxAbsDiff", "bestUplift"):
            raise ValueError(f"rule must be 'maxAbsDiff' or 'bestUplift'; got {rule!r}")

        X = check_array(X, dtype=DTYPE, accept_sparse="csc")
        treatment = np.asarray(treatment)
        y = (np.asarray(y).ravel() > 0).astype(np.float64)
        t_idx = np.fromiter(
            (self._group2index[t] for t in treatment), dtype=int, count=len(treatment)
        )

        tree = self.tree_
        n_nodes = tree.node_count
        left = tree.children_left
        right = tree.children_right
        is_orig_leaf = left == -1
        n_groups = self.n_outputs_

        # Build-time (training) per-group P(Y=1|T=g) at every node -- the raw
        # leaf estimate the kernel stores in ``value``; a collapsed node falls
        # back to this, mirroring legacy ``backupResults``.
        train_p = tree.value[:, :, 0]  # (n_nodes, n_groups)
        diffs = train_p[:, 1:] - train_p[:, [0]]  # (n_nodes, n_treatments)
        rows = np.arange(n_nodes)
        max_diff_t = np.abs(diffs).argmax(axis=1) + 1  # group idx of max |uplift|
        max_diff_sign = np.sign(diffs[rows, max_diff_t - 1])
        best_t = diffs.argmax(axis=1) + 1  # group idx of best positive uplift
        best_t = np.where(diffs[rows, best_t - 1] > 0, best_t, 0)  # 0 == control

        # Validation per-node, per-group counts and positive-outcome rates.
        dpath = tree.decision_path(X)  # sparse (n_val, n_nodes)
        n_val = np.zeros((n_nodes, n_groups))
        p_val = np.zeros((n_nodes, n_groups))
        for g in range(n_groups):
            in_g = t_idx == g
            if not in_g.any():
                continue
            dp_g = dpath[in_g]
            counts = np.asarray(dp_g.sum(axis=0)).ravel()
            pos = np.asarray(dp_g[y[in_g] > 0].sum(axis=0)).ravel()
            n_val[:, g] = counts
            p_val[:, g] = np.divide(
                pos, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0
            )

        def uplift(node, treat_idx, sign):
            """Signed validation uplift of ``treat_idx`` vs control at ``node``."""
            return sign * (p_val[node, treat_idx] - p_val[node, 0])

        def keep_split(node):
            """True if the split at ``node`` should be kept (not pruned)."""
            lft, rgt = left[node], right[node]
            if rule == "maxAbsDiff":
                cur = uplift(node, max_diff_t[node], max_diff_sign[node])
                child = 0.0
                for c in (lft, rgt):
                    mt = max_diff_t[c]
                    denom = n_val[node, mt] + n_val[node, 0]
                    if denom <= 0:
                        continue
                    weight = (n_val[c, mt] + n_val[c, 0]) / denom
                    child += uplift(c, mt, max_diff_sign[c]) * weight
                gain = child - cur
            else:  # bestUplift
                cur = uplift(node, best_t[node], 1.0)
                total = n_val[lft].sum() + n_val[rgt].sum()
                if total <= 0:
                    return False
                child = n_val[lft].sum() / total * uplift(
                    lft, best_t[lft], 1.0
                ) + n_val[rgt].sum() / total * uplift(rgt, best_t[rgt], 1.0)
                gain = child - cur
            return gain > minGain and child >= 0.0

        # Bottom-up merge: repeatedly collapse internal nodes whose children are
        # both leaves-now and whose split is not worth keeping, until stable.
        collapsed = np.zeros(n_nodes, dtype=bool)
        changed = True
        while changed:
            changed = False
            for node in range(n_nodes):
                if is_orig_leaf[node] or collapsed[node]:
                    continue
                lft, rgt = left[node], right[node]
                leaf_now_l = is_orig_leaf[lft] or collapsed[lft]
                leaf_now_r = is_orig_leaf[rgt] or collapsed[rgt]
                if leaf_now_l and leaf_now_r and not keep_split(node):
                    collapsed[node] = True
                    changed = True

        if not collapsed.any():
            return self

        # Rebuild a compact tree: every surviving leaf (original leaves +
        # collapsed internal nodes) is marked in the mask.
        leaves_in_subtree = (is_orig_leaf | collapsed).astype(np.uint8)
        pruned = Tree(self.n_features_, self.tree_.n_classes, self.n_outputs_)
        build_pruned_tree_from_mask(pruned, self.tree_, leaves_in_subtree)
        self.tree_ = pruned
        return self

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
