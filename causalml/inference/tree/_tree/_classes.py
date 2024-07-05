"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause


from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import MultiOutputMixin
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier, clone
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

import numpy as np
from scipy.sparse import issparse

from ._tree import Tree
from ._tree import _build_pruned_tree_ccp
from ._tree import ccp_pruning_path
from . import _tree, _splitter

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

DENSE_SPLITTERS = {
    "best": _splitter.BestSplitter,
    "random": _splitter.RandomSplitter,
}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter,
}


# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    @abstractmethod
    def fit(
        self,
        X,
        treatment,
        y,
        sample_weight=None,
        check_input=True,
        X_idx_sorted="deprecated",
    ):
        pass

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
            if issparse(X) and (
                X.indices.dtype != np.intc or X.indptr.dtype != np.intc
            ):
                raise ValueError("No support for np.int64 index based sparse matrices")
        else:
            # The number of features is checked regardless of `check_input`
            self._check_n_features(X, reset=False)
        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # Classification
        if is_classifier(self):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                class_type = self.classes_[0].dtype
                predictions = np.zeros((n_samples, self.n_outputs_), dtype=class_type)
                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1), axis=0
                    )

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    def _prune_tree(self):
        """Prune tree using Minimal Cost-Complexity Pruning."""
        check_is_fitted(self)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if self.ccp_alpha == 0.0:
            return

        # build pruned tree
        if is_classifier(self):
            n_classes = np.atleast_1d(self.n_classes_)
            pruned_tree = Tree(self.n_features_in_, n_classes, self.n_outputs_)
        else:
            pruned_tree = Tree(
                self.n_features_in_,
                # TODO: the tree shouldn't need this param
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )
        _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

        self.tree_ = pruned_tree

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """Compute the pruning path during Minimal Cost-Complexity Pruning.

        See :ref:`minimal_cost_complexity_pruning` for details on the pruning
        process.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        ccp_path : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            ccp_alphas : ndarray
                Effective alphas of subtree during pruning.

            impurities : ndarray
                Sum of the impurities of the subtree leaves for the
                corresponding alpha value in ``ccp_alphas``.
        """
        est = clone(self).set_params(ccp_alpha=0.0)
        est.fit(X, y, sample_weight=sample_weight)
        return Bunch(**ccp_pruning_path(est.tree_))

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).
        """
        check_is_fitted(self)

        return self.tree_.compute_feature_importances()
