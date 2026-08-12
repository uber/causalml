import copy
import numbers
import warnings
from math import ceil
from typing import Union

try:
    from packaging.version import parse as Version
except ModuleNotFoundError:
    from distutils.version import LooseVersion as Version

import numpy as np
from scipy.sparse import issparse
from sklearn import __version__ as sklearn_version
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight, validate_data

from .._tree._classes import DTYPE, DOUBLE
from .._tree._classes import SPARSE_SPLITTERS, DENSE_SPLITTERS
from .._tree._classes import Tree, BaseDecisionTree
from .._tree._criterion import Criterion
from .._tree._splitter import Splitter

from ._builder import DepthFirstCausalTreeBuilder, BestFirstCausalTreeBuilder
from ._criterion import StandardMSE, CausalMSE, TTest

CAUSAL_TREES_CRITERIA = {
    "causal_mse": CausalMSE,
    "standard_mse": StandardMSE,
    "t_test": TTest,
}


def get_check_y_params() -> dict:
    """
    Prepares flags for sklearn 1.6+.

    Returns: check_y_params
    """
    check_y_params = dict(ensure_2d=False, dtype=None, ensure_all_finite=False)
    return check_y_params


class BaseCausalDecisionTree(BaseDecisionTree):
    """
    Modified base class BaseDecisionTree for causal trees
    Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py
    """

    def __init__(self, min_group_samples: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_group_samples = min_group_samples

    def _support_missing_values(self, X) -> bool:
        """Whether native missing-value (NaN) splitting is available for ``X``.

        Mirrors scikit-learn's ``BaseDecisionTree._support_missing_values``. The
        causal criterion implements scikit-learn's per-feature missing-value
        accumulation, which the dense best splitter threads through (see
        ``__sklearn_tags__``); NaNs are rejected for sparse input or a non-default
        splitter.
        """
        return (
            not issparse(X)
            and self.__sklearn_tags__().input_tags.allow_nan
            and self.monotonic_cst is None
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self.splitter == "best"
        return tags

    def _prune_tree(self):
        """Prune with the fitted ``ccp_alpha_`` when one was selected.

        Mirrors the base implementation, which reads the ``ccp_alpha`` constructor
        parameter directly. ``CausalTreeRegressor`` may choose the penalty by
        cross-validation instead (``ccp_alpha="cv"``), and sklearn's convention
        is that a value learned at fit time lives on a trailing-underscore attribute
        rather than overwriting the parameter.
        """
        ccp_alpha = getattr(self, "ccp_alpha_", None)
        if ccp_alpha is None or ccp_alpha == self.ccp_alpha:
            return super()._prune_tree()

        original, self.ccp_alpha = self.ccp_alpha, ccp_alpha
        try:
            return super()._prune_tree()
        finally:
            self.ccp_alpha = original

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Union[np.ndarray, None] = None,
        check_input: bool = True,
        X_idx_sorted="deprecated",
    ):
        random_state = check_random_state(self.random_state)

        # ``ccp_alpha`` may be a sentinel such as "cv", in which case the resolved
        # penalty lives on ``ccp_alpha_`` and the sentinel itself is not comparable.
        ccp_alpha = getattr(self, "ccp_alpha_", self.ccp_alpha)
        if not isinstance(ccp_alpha, str) and ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        missing_values_in_feature_mask = None
        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be csr.
            check_X_params = dict(
                dtype=DTYPE, accept_sparse="csc", ensure_all_finite=False
            )
            check_y_params = get_check_y_params()
            X, y = validate_data(
                self, X, y, validate_separately=(check_X_params, check_y_params)
            )
            # Native missing-value support (scikit-learn convention): rejects inf
            # and, when NaNs are not supported, rejects them too; otherwise returns
            # the per-feature missing mask threaded to the builder.
            missing_values_in_feature_mask = (
                self._compute_missing_values_in_feature_mask(X)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based " "sparse matrices"
                    )

            if self.criterion not in CAUSAL_TREES_CRITERIA.keys():
                raise ValueError(
                    f"Only {CAUSAL_TREES_CRITERIA.keys()} criteria are supported"
                )

        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_

        y = np.atleast_1d(y)
        expanded_class_weight = None

        # n_outputs_ is the length of [y|control, y|treatment_1,..., y|treatment_{n-1}]
        self.n_outputs_ = y.shape[1]

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0.0 < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s" % self.min_samples_split
                )
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0.0 < self.min_samples_split <= 1.0:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s" % self.min_samples_split
                )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match "
                "number of samples=%d" % (len(y), n_samples)
            )
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError(
                "max_leaf_nodes must be integral number but was " "%r" % max_leaf_nodes
            )
        if -1 < max_leaf_nodes < 2:
            raise ValueError(
                ("max_leaf_nodes {0} must be either None " "or larger than 1").format(
                    max_leaf_nodes
                )
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        if X_idx_sorted != "deprecated":
            warnings.warn(
                "The parameter 'X_idx_sorted' is deprecated and has no "
                "effect. It will be removed in 1.1 (renaming of 0.26). You "
                "can suppress this warning by not passing any value to the "
                "'X_idx_sorted' parameter.",
                FutureWarning,
            )

        # Build tree
        criterion = self.criterion
        if isinstance(criterion, str):
            criterion = CAUSAL_TREES_CRITERIA[criterion](self.n_outputs_, n_samples)
            criterion.groups_penalty = self.groups_penalty
            # N^tr / N^est, scaling the honest variance penalty. Read via getattr
            # so subclasses that do not do honest estimation keep the unscaled
            # penalty (0.0) without having to declare the attribute.
            criterion.train_to_est_ratio = getattr(self, "_train_to_est_ratio", 0.0)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst=None,
            )
            self.tree_ = Tree(
                self.n_features_,
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstCausalTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
                self.min_group_samples,
            )
        else:
            builder = BestFirstCausalTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
                self.min_group_samples,
            )
        # Treatment column is described via y cols. The first column is always a control group.
        builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)

        self._prune_tree()

        return self
