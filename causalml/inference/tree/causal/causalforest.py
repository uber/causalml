from typing import Union

import numpy as np
import forestci as fci
from joblib import Parallel, delayed
from warnings import catch_warnings, simplefilter, warn

from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_random_state, _check_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn import __version__ as sklearn_version
from sklearn.ensemble._forest import DOUBLE, DTYPE, MAX_INT
from sklearn.ensemble._forest import ForestRegressor
from sklearn.ensemble._forest import compute_sample_weight, issparse
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap

from .causaltree import CausalTreeRegressor

try:
    from packaging.version import parse as Version
except ModuleNotFoundError:
    from distutils.version import LooseVersion as Version

if Version(sklearn_version) >= Version("1.1.0"):
    _joblib_parallel_args = dict(prefer="threads")
else:
    from sklearn.utils.fixes import _joblib_parallel_args

    _joblib_parallel_args = _joblib_parallel_args(prefer="threads")


def _parallel_build_trees(
    tree,
    forest,
    X,
    treatment,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree.fit(X, treatment, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, treatment, y, sample_weight=sample_weight, check_input=False)

    return tree


class CausalRandomForestRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators: int = 100,
        *,
        control_name: Union[int, str] = 0,
        criterion: str = "causal_mse",
        alpha: float = 0.05,
        max_depth: int = None,
        min_samples_split: int = 60,
        min_samples_leaf: int = 100,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str] = 1.0,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = float("-inf"),
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        groups_penalty: float = 0.5,
        max_samples: int = None,
        groups_cnt: bool = True,
    ):
        """
        Initialize Random Forest of CausalTreeRegressors

        Args:
            n_estimators: (int, default=100)
                    Number of trees in the forest
            control_name: (str or int)
                    Name of control group
            criterion: ({"causal_mse", "standard_mse"}, default="causal_mse"):
                    Function to measure the quality of a split.
            alpha: (float)
                    The confidence level alpha of the ATE estimate and ITE bootstrap estimates
            max_depth: (int, default=None)
                    The maximum depth of the tree.
            min_samples_split: (int or float, default=2)
                The minimum number of samples required to split an internal node:
            min_samples_leaf: (int or float), default=100
                The minimum number of samples required to be at a leaf node.
            min_weight_fraction_leaf: (float, default=0.0)
                The minimum weighted fraction of the sum total of weights (of all
                the input samples) required to be at a leaf node.
            max_features: (int, float or {"auto", "sqrt", "log2"}, default=None)
                The number of features to consider when looking for the best split
            max_leaf_nodes: (int, default=None)
                Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            min_impurity_decrease: (float, default=float("-inf")))
                A node will be split if this split induces a decrease of the impurity
                greater than or equal to this value.
            bootstrap : (bool, default=True)
                Whether bootstrap samples are used when building trees.
            oob_score : bool, default=False
                Whether to use out-of-bag samples to estimate the generalization score.
            n_jobs : int, default=None
                    The number of jobs to run in parallel.
            random_state : (int, RandomState instance or None, default=None)
                    Controls both the randomness of the bootstrapping of the samples used
                    when building trees (if ``bootstrap=True``) and the sampling of the
                    features to consider when looking for the best split at each node
                    (if ``max_features < n_features``).
            verbose : (int, default=0)
                    Controls the verbosity when fitting and predicting.
            warm_start : (bool, default=False)
                    When set to ``True``, reuse the solution of the previous call to fit
                    and add more estimators to the ensemble, otherwise, just fit a whole
                    new forest.
            ccp_alpha : (non-negative float, default=0.0)
                    Complexity parameter used for Minimal Cost-Complexity Pruning.
            groups_penalty: (float, default=0.5)
                    This penalty coefficient manages the node impurity increase in case of the difference between
                    treatment and control samples sizes.
            max_samples : (int or float, default=None)
                    If bootstrap is True, the number of samples to draw from X
                    to train each base estimator.
            groups_cnt: (bool), count treatment and control groups for each node/leaf
        """
        self._estimator = CausalTreeRegressor(
            control_name=control_name, criterion=criterion, groups_cnt=groups_cnt
        )
        _estimator_key = (
            "estimator"
            if Version(sklearn_version) >= Version("1.2.0")
            else "base_estimator"
        )
        _parent_args = {
            _estimator_key: self._estimator,
            "n_estimators": n_estimators,
            "estimator_params": (
                "criterion",
                "control_name",
                "max_depth",
                "min_samples_split",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "ccp_alpha",
                "groups_penalty",
                "min_samples_leaf",
                "random_state",
            ),
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "max_samples": max_samples,
        }

        super().__init__(**_parent_args)

        self.criterion = criterion
        self.control_name = control_name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.groups_penalty = groups_penalty
        self.alpha = alpha
        self.groups_cnt = groups_cnt

    def _fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        """
        Build a forest of trees from the training set (X, y).
        With modified _parallel_build_trees for Causal Trees used in BaseForest.fit()
        Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        treatment : array-like of shape (n_samples,)
            The treatment assignments.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.max_outputs_ = np.unique(treatment).astype(int).size + 1
        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                **_joblib_parallel_args,
            )(
                delayed(_parallel_build_trees)(
                    tree=t,
                    forest=self,
                    X=X,
                    treatment=treatment,
                    y=y,
                    sample_weight=sample_weight,
                    tree_idx=i,
                    n_trees=len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            self.estimators_.extend(trees)

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        """
        Fit Causal RandomForest
        Args:
            X: (np.ndarray), feature matrix
            treatment: (np.ndarray), treatment vector
            y: (np.ndarray), outcome vector
            sample_weight: (np.ndarray), sample weights
        Returns:
             self
        """
        X, y, w = self._estimator._prepare_data(X=X, treatment=treatment, y=y)
        return self._fit(X=X, treatment=w, y=y, sample_weight=sample_weight)

    def predict(self, X: np.ndarray, with_outcomes: bool = False) -> np.ndarray:
        """Predict individual treatment effects

        Args:
            X (np.matrix): a feature matrix
            with_outcomes (bool), default=False,
                                  include outcomes Y_hat(X|T=0), Y_hat(X|T=1) along with individual treatment effect
        Returns:
           (np.matrix): individual treatment effect (ITE), dim=nx1
                        or ITE with outcomes [Y_hat(X|T=0), Y_hat(X|T=1), ITE], dim=nx3
        """
        if with_outcomes:
            self.n_outputs_ = self.max_outputs_
            for estimator in self.estimators_:
                estimator._with_outcomes = True
        else:
            self.n_outputs_ = 1
        y_pred = super().predict(X)
        return y_pred

    def calculate_error(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        inbag: np.ndarray = None,
        calibrate: bool = True,
        memory_constrained: bool = False,
        memory_limit: int = None,
    ) -> np.ndarray:
        """
        Calculate error bars from scikit-learn RandomForest estimators
        Source:
        https://github.com/scikit-learn-contrib/forest-confidence-interval

        Args:
            X_train: (np.ndarray), training subsample of feature matrix, (n_train_sample, n_features)
            X_test: (np.ndarray), test subsample of feature matrix, (n_train_sample, n_features)
            inbag: (ndarray, optional),
                    The inbag matrix that fit the data. If set to `None` (default) it
                    will be inferred from the forest. However, this only works for trees
                    for which bootstrapping was set to `True`. That is, if sampling was
                    done with replacement. Otherwise, users need to provide their own
                    inbag matrix.
            calibrate: (boolean, optional)
                    Whether to apply calibration to mitigate Monte Carlo noise.
                    Some variance estimates may be negative due to Monte Carlo effects if
                    the number of trees in the forest is too small. To use calibration,
                    Default: True
            memory_constrained: (boolean, optional)
                    Whether or not there is a restriction on memory. If False, it is
                    assumed that a ndarray of shape (n_train_sample,n_test_sample) fits
                    in main memory. Setting to True can actually provide a speedup if
                    memory_limit is tuned to the optimal range.
            memory_limit: (int, optional)
                    An upper bound for how much memory the intermediate matrices will take
                    up in Megabytes. This must be provided if memory_constrained=True.

        Returns:
            (np.ndarray), An array with the unbiased sampling variance for a RandomForest object.
        """

        var = fci.random_forest_error(
            self,
            X_train,
            X_test,
            inbag=inbag,
            calibrate=calibrate,
            memory_constrained=memory_constrained,
            memory_limit=memory_limit,
        )
        return var
