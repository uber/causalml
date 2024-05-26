import logging
from typing import Union

import tqdm
import numpy as np
from numpy import float32 as DTYPE

from pathos.pools import ProcessPool as PPool
from scipy.stats import norm
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from causalml.inference.meta.utils import check_treatment_vector

from ._tree import BaseCausalDecisionTree
from ..utils import get_tree_leaves_mask, timeit

logger = logging.getLogger("causalml")


class CausalTreeRegressor(RegressorMixin, BaseCausalDecisionTree):
    """A Causal Tree regressor class.
    The Causal Tree is a decision tree regressor with a split criteria for treatment effects.
    Details are available at `Athey and Imbens (2015) <https://arxiv.org/abs/1504.01132)>`_.
    """

    def __init__(
        self,
        *,
        criterion: str = "causal_mse",
        splitter: str = "best",
        alpha: float = 0.05,
        control_name: Union[int, str] = 0,
        max_depth: int = None,
        min_samples_split: Union[int, float] = 60,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str] = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = float("-inf"),
        ccp_alpha: float = 0.0,
        groups_penalty: float = 0.5,
        min_samples_leaf: int = 100,
        random_state: int = None,
        groups_cnt: bool = False,
        groups_cnt_mode: str = "nodes",
    ):
        """
        Initialize a Causal Tree
        Args:
            criterion: ({"causal_mse", "standard_mse"}, default="causal_mse")
                The function to measure the quality of a split.
            splitter: ({"best", "random"}, default="best")
                The strategy used to choose the split at each node. Supported
                strategies are "best" to choose the best split and "random" to choose
                the best random split.
            alpha: (float): the confidence level alpha of the ATE estimate and ITE bootstrap estimates
            control_name: (str or int): name of control group
            max_depth: (int, default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure or until all leaves contain less than
                min_samples_split samples.
            min_samples_split: (int or float, default=2)
                The minimum number of samples required to split an internal node:
                - If int, then consider `min_samples_split` as the minimum number.
                - If float, then `min_samples_split` is a fraction and
                  `ceil(min_samples_split * n_samples)` are the minimum
                  number of samples for each split.
            min_weight_fraction_leaf: (float, default=0.0)
                The minimum weighted fraction of the sum total of weights (of all
                the input samples) required to be at a leaf node. Samples have
                equal weight when sample_weight is not provided.
            max_features: (int, float or {"auto", "sqrt", "log2"}, default=None)
                The number of features to consider when looking for the best split:

                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `int(max_features * n_features)` features are considered at each
                  split.
                - If "auto", then `max_features=n_features`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.
            max_leaf_nodes: (int, default=None)
                Grow a tree with ``max_leaf_nodes`` in best-first fashion.
                Best nodes are defined as relative reduction in impurity.
                If None then unlimited number of leaf nodes.
            min_impurity_decrease: (float, default=float("-inf")))
                A node will be split if this split induces a decrease of the impurity
                greater than or equal to this value.
            ccp_alpha: (non-negative float, default=0.0)
                Complexity parameter used for Minimal Cost-Complexity Pruning. The
                subtree with the largest cost complexity that is smaller than
                ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
                :ref:`minimal_cost_complexity_pruning` for details.
            groups_penalty: (float, default=0.5)
                This penalty coefficient manages the node impurity increase in case of the difference between
                treatment and control samples sizes.
            min_samples_leaf: (int or float), default=100
                The minimum number of samples required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at
                least ``min_samples_leaf`` training samples in each of the left and
                right branches.  This may have the effect of smoothing the model,
                especially in regression.

                - If int, then consider `min_samples_leaf` as the minimum number.
                - If float, then `min_samples_leaf` is a fraction and
                  `ceil(min_samples_leaf * n_samples)` are the minimum
                  number of samples for each node.
            random_state: (int), RandomState instance or None, default=None
                Used to pick randomly the `max_features` used at each split.
                See :term:`Glossary <random_state>` for details.
            groups_cnt: (bool), count treatment and control groups for each node/leaf
            groups_cnt_mode: (str, 'nodes', 'leaves'), mode for samples counting
        """

        self.criterion = criterion
        self.splitter = splitter
        self.alpha = alpha
        self.control_name = control_name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.groups_penalty = groups_penalty
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self._classes = {}
        self.groups_cnt = groups_cnt
        self.groups_cnt_mode = groups_cnt_mode
        self._with_outcomes = False
        self._groups_cnt = {}

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
        check_input=False,
    ):
        """
        Fit CausalTreeRegressor
        Args:
            X (np.ndarray): feature matrix
            treatment (np.ndarray): treatment vector
            y (np.ndarray): outcome vector
            sample_weight (np.ndarray): sample_weight
            check_input (bool, optional): default=False
        Returns:
            self
        """

        if self.criterion == "causal_mse" and self.min_impurity_decrease != float(
            "-inf"
        ):
            raise ValueError(
                "min_impurity_decrease must be set to -inf for causal_mse criterion"
            )

        X, y, w = self._prepare_data(X=X, y=y, treatment=treatment)
        self.treatment_groups = np.unique(w)

        super().fit(
            X=X, treatment=w, y=y, sample_weight=sample_weight, check_input=check_input
        )

        if self.groups_cnt:
            self._groups_cnt = self._count_groups_distribution(X=X, treatment=w)
        return self

    def predict(
        self, X: np.ndarray, with_outcomes: bool = False, check_input=True
    ) -> np.ndarray:
        """Predict individual treatment effects

        Args:
            X (np.matrix): a feature matrix
            with_outcomes (bool), default=False,
                                  include outcomes Y_hat(X|T=0), Y_hat(X|T=1) along with individual treatment effect
            check_input (bool), default=True,
                                Allow to bypass several input checking.
        Returns:
           (np.matrix): individual treatment effect (ITE), dim=nx1
                        or ITE with outcomes [Y_hat(X|T=0), Y_hat(X|T=1), ITE], dim=nx3
        """
        if check_input:
            X = self._validate_X_predict(X, check_input)
        y_outcomes = super().predict(X)
        y_pred = y_outcomes[:, 1] - y_outcomes[:, 0]
        need_outcomes = with_outcomes or self._with_outcomes
        return (
            np.hstack([y_outcomes, y_pred.reshape(-1, 1)]) if need_outcomes else y_pred
        )

    def fit_predict(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        return_ci: bool = False,
        n_bootstraps: int = 1000,
        bootstrap_size: int = 10000,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> tuple:
        """Fit the Causal Tree model and predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            n_jobs (int): the number of jobs for bootstrap
            verbose (str): whether to output progress logs

        Returns:
           (tuple):

             - te (numpy.ndarray): Predictions of treatment effects.
             - te_lower (numpy.ndarray, optional): lower bounds of treatment effects
             - te_upper (numpy.ndarray, optional): upper bounds of treatment effects
        """
        self.fit(X=X, treatment=treatment, y=y)
        te = self.predict(X=X)

        if return_ci:
            te_bootstraps = self.bootstrap_pool(
                X=X,
                treatment=treatment,
                y=y,
                n_bootstraps=n_bootstraps,
                bootstrap_size=bootstrap_size,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            te_lower = np.percentile(te_bootstraps, (self.alpha / 2) * 100, axis=0)
            te_upper = np.percentile(te_bootstraps, (1 - self.alpha / 2) * 100, axis=0)
            return te, te_lower, te_upper
        else:
            return te

    def estimate_ate(
        self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Estimate the Average Treatment Effect (ATE).
        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        Returns:
            tuple, The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        dhat = self.fit_predict(X, treatment, y)

        te = dhat.mean()
        se = dhat.std() / X.shape[0]

        te_lb = te - se * norm.ppf(1 - self.alpha / 2)
        te_ub = te + se * norm.ppf(1 - self.alpha / 2)

        return te, te_lb, te_ub

    @timeit(exclude_kwargs=("X", "treatment", "y"))
    def bootstrap_pool(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        n_bootstraps: int,
        bootstrap_size: int,
        n_jobs: int,
        verbose: bool,
    ):
        """
        Run a pool of bootstraps
        Args:
            X (np.ndarray):  a feature matrix
            treatment (np.ndarray): a treatment vector
            y (np.ndarray): an outcome vector
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            n_jobs (int): number of processes
            verbose (bool): whether to output progress logs

        Returns:
            (np.ndarray), bootstrap estimates

        """

        def _bootstrap(i: int):
            if verbose:
                logger.info(f"Boostrap iteration: {i}")
            return self.bootstrap(
                X=X, treatment=treatment, y=y, sample_size=bootstrap_size, seed=i
            )

        pool = PPool(nodes=n_jobs)
        pool.restart(force=True)

        bootstrap_estimates = np.array(
            list(
                tqdm.tqdm(
                    pool.imap(_bootstrap, (i for i in range(n_bootstraps))),
                    total=n_bootstraps,
                )
            )
        )
        pool.close()
        pool.join()
        return bootstrap_estimates

    def bootstrap(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_size: int,
        seed: int,
    ) -> np.ndarray:
        """Runs a single bootstrap.

        Fits on bootstrapped sample, then predicts on whole population.

        Args:
            X (np.ndarray): a feature matrix
            treatment (np.ndarray): a treatment vector
            y (np.ndarray): an outcome vector
            sample_size (int): bootstrap sample size
            seed: (int): bootstrap seed

        Returns:
            (np.ndarray): bootstrap predictions
        """
        _rnd = np.random.RandomState(seed=seed)
        idxs = _rnd.choice(np.arange(0, X.shape[0]), size=sample_size)
        X_b, y_b, treatment_b = X[idxs], y[idxs], treatment[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X)
        return te_b

    def _prepare_data(
        self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray
    ) -> tuple:
        """
        Prepare input data with treatment info for DecisionTreeRegressor
        Args:
            X: : (np.ndarray), feature matrix
            treatment: : (np.ndarray), treatment vector
            y: : (np.ndarray), outcome vector
        Returns: X, y, w
        """
        if y.shape[0] != treatment.shape[0]:
            raise ValueError(
                f"The number of `treatment` and `y` rows are not equal: {y.shape[0]} {treatment.shape[0]}"
            )
        check_treatment_vector(treatment, self.control_name)

        self.is_treatment = treatment != self.control_name
        w = self.is_treatment.astype(int)

        X = check_array(X, dtype=DTYPE, accept_sparse="csc")
        y = check_array(y, ensure_2d=False, dtype=None)

        self.n_samples, self.n_features = X.shape

        return X, y, w

    def _count_groups_distribution(self, X: np.ndarray, treatment: np.ndarray) -> dict:
        """
        Count treatment, control distribution for tree nodes/leaves
        Args:
            X: (np.ndarray), feature matrix
            treatment: (np.ndarray), treatment vector
        Returns:
            dict: treatment groups for each tree node/leaves
        """
        check_is_fitted(self)

        self.is_leaves = get_tree_leaves_mask(self)
        groups_cnt = {
            idx: {group: 0 for group in self.treatment_groups}
            for idx in np.array(range(self.tree_.node_count))
        }
        node_indicators = self.tree_.decision_path(X.astype(np.float32))

        for sample_id in range(X.shape[0]):
            nodes_path = node_indicators.indices[
                node_indicators.indptr[sample_id] : node_indicators.indptr[
                    sample_id + 1
                ]
            ]

            if self.groups_cnt_mode == "leaves":
                groups_cnt[nodes_path[-1]][treatment[sample_id]] += 1
            elif self.groups_cnt_mode == "nodes":
                for node_id in nodes_path:
                    groups_cnt[node_id][treatment[sample_id]] += 1
        return groups_cnt
