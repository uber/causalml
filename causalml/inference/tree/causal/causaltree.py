import logging
from typing import Union

import tqdm
import numpy as np
from numpy import float32 as DTYPE

from pathos.pools import ProcessPool as PPool
from scipy.stats import norm, ttest_ind
from sklearn.base import RegressorMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from causalml.inference.meta.utils import check_treatment_vector
from causalml.inference.serialization import SerializableLearner

from ._tree import BaseCausalDecisionTree
from .._tree._tree import Tree, _build_pruned_tree_ccp, ccp_pruning_path
from ..utils import _check_fraction, get_tree_leaves_mask, timeit

logger = logging.getLogger("causalml")

#: Cap on the cost-complexity candidates scored per fold. The path can hold one
#: alpha per internal node, and scoring is cheap but not free.
MAX_CCP_CANDIDATES = 40

#: ``ccp_alpha`` sentinel selecting the penalty by cross-validation.
CV_PENALTY = "cv"


class CausalTreeRegressor(SerializableLearner, RegressorMixin, BaseCausalDecisionTree):
    """A Causal Tree regressor class.
    The Causal Tree is a decision tree regressor with a split criteria for treatment effects.
    Details are available at `Athey and Imbens (2015) <https://arxiv.org/abs/1504.01132)>`_.

    .. note::
        **Observational data needs inverse-propensity weights.** Every criterion here
        compares raw group means, with no adjustment for how treatment was assigned, so
        when the propensity varies with ``X`` both the split search and the leaf estimates
        inherit the confounding bias. Pass inverse-propensity weights as ``sample_weight``
        to correct it::

            e_hat = cross_val_predict(clf, X, treatment, cv=5, method="predict_proba")[:, 1]
            e_hat = np.clip(e_hat, 0.05, 0.95)
            ipw = np.where(treatment == 1, 1 / e_hat, 1 / (1 - e_hat))
            model.fit(X=X, treatment=treatment, y=y, sample_weight=ipw)

        Measured on Nie & Wager Setup A over 10 seeds (a confounded design whose propensity
        and treatment effect share the same features), for
        :class:`CausalRandomForestRegressor` with a cross-fitted propensity:

        ==================  ==================  =========  =========
        weights             corr with true tau  ATE error  CATE RMSE
        ==================  ==================  =========  =========
        none                0.381               0.389      0.433
        inverse-propensity  0.844               0.067      0.142
        ==================  ==================  =========  =========

        No correction is needed for a randomized experiment. See
        ``docs/examples/causal_tree_honesty_parity.ipynb`` for the full comparison.
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
        ccp_alpha: Union[float, str] = 0.0,
        groups_penalty: float = 0.5,
        min_group_samples: int = 50,
        min_samples_leaf: int = 100,
        random_state: int = None,
        groups_cnt: bool = False,
        groups_cnt_mode: str = "nodes",
        node_pvalues: bool = False,
        honesty: bool = True,
        estimation_sample_size: float = 0.5,
        cv_folds: int = 5,
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
            control_name: (str or int): name or index of control group
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
            ccp_alpha: (non-negative float or "cv", default=0.0)
                Complexity parameter used for Minimal Cost-Complexity Pruning. The
                subtree with the largest cost complexity that is smaller than
                ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
                :ref:`minimal_cost_complexity_pruning` for details.

                ``"cv"`` selects it by ``cv_folds``-fold cross-validation, completing the
                CT-H algorithm of `Athey and Imbens (2016)
                <https://arxiv.org/abs/1504.01132>`_. ``honesty=True`` supplies held-out
                leaf estimation; ``"cv"`` adds the two remaining pieces:

                1. The splitting objective's variance penalty is scaled by
                   ``1 + N_structure / N_estimation`` (the paper's factor of 2 at an even
                   split), pricing the noise the held-out leaf estimates will carry.
                2. Every subtree on the cost-complexity path is scored with that same
                   objective evaluated on the held-out fold -- the paper's
                   ``-EMSE_tau(S^tr,cv, Pi)`` -- and the best-scoring penalty is used.
                   Otherwise the tree grows until ``min_samples_leaf`` /
                   ``min_group_samples`` stop it, and the variance penalty only ranks
                   candidate splits rather than choosing tree size, which is the job it
                   does in the paper.

                Requires ``honesty=True``, and costs ``cv_folds`` extra fits. Held-out CATE
                RMSE over 10 paired seeds fell by 25% at ``sigma=0.5`` and 55% at
                ``sigma=2.0`` (10 of 10 seeds each), and is unchanged where there is no
                overfitting to remove. This applies to a single tree; on
                :class:`CausalRandomForestRegressor` it measured no gain and is worse at
                low noise, because averaging across trees already removes that variance.
            groups_penalty: (float, default=0.5)
                This penalty coefficient manages the node impurity increase in case of the difference between
                treatment and control samples sizes.
            min_group_samples: (int, default=50)
                The minimum number of samples per each group: k treatment groups and control group.
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
            node_pvalues: (bool), compute treatment effect p-values for each node.
                Note: These are naive in-sample t-tests and do not account for the tree
                structure or multiple testing. They should be used for descriptive
                purposes only and are not valid for post-selection inference. They are
                computed on the full ``fit`` sample even when ``honesty=True``.
            honesty: (bool, default=True), use the honest approach of
                `Athey and Imbens (2016) <https://arxiv.org/abs/1504.01132>`_: split the
                sample in two, grow the tree structure on one half and re-estimate each
                leaf's group outcome means on the other. The leaf estimates are then
                independent of the splits that produced them, which removes the
                overfitting bias of a tree that both chooses and estimates on the same
                rows. The cost is that each half sees only part of the data, so an honest
                tree is usually shallower and noisier per leaf.

                On by default, matching ``grf``'s ``honesty = TRUE`` and EconML's
                ``honest=True``. Pass ``honesty=False`` for the pre-0.18 behavior, where
                the same rows both chose the splits and estimated the leaves.

                Two details differ from ``grf``: the split here is stratified on
                treatment, so both halves keep every group; and a leaf that the
                estimation half leaves empty keeps its structure-half value instead of
                being pruned into its sibling (``honesty.prune.leaves``).
            estimation_sample_size: (float, default=0.5), fraction of the sample held out
                for leaf re-estimation when ``honesty=True``. Ignored otherwise.
            cv_folds: (int, default=5), folds used to select the penalty when
                ``ccp_alpha="cv"``. Ignored otherwise.
        """

        self.criterion = criterion
        self.splitter = splitter
        self.alpha = alpha
        self.control_name = control_name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_group_samples = min_group_samples
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.groups_penalty = groups_penalty
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.honesty = honesty
        self.estimation_sample_size = estimation_sample_size
        self.cv_folds = cv_folds

        self._classes = {}
        self.groups_cnt = groups_cnt
        self.groups_cnt_mode = groups_cnt_mode
        self.node_pvalues = node_pvalues
        self._with_outcomes = False
        self._groups_cnt = {}
        self._node_pvalues = {}

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_group_samples=min_group_samples,
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
        sample_weight: Union[np.ndarray, None] = None,
        check_input: bool = True,
        prepare_data: bool = True,
    ):
        """
        Fit CausalTreeRegressor

        Args:
            X (np.ndarray): feature matrix
            treatment (np.ndarray): treatment vector, includes control group
            y (np.ndarray): outcome vector
            sample_weight (np.ndarray): sample_weight, optional. Weights the split search
                and the leaf estimates alike, including the honest re-estimation, so
                inverse-propensity weights passed here correct the confounding bias on
                observational data — see the note on the class.
            check_input (bool, optional): default=False
            prepare_data (bool): default=True

        Returns:
            self
        """

        if self.criterion == "causal_mse" and self.min_impurity_decrease != float(
            "-inf"
        ):
            raise ValueError(
                "min_impurity_decrease must be set to -inf for causal_mse criterion"
            )

        _check_fraction("estimation_sample_size", self.estimation_sample_size)

        if isinstance(self.ccp_alpha, str):
            if self.ccp_alpha != CV_PENALTY:
                raise ValueError(
                    f"ccp_alpha must be a non-negative float or {CV_PENALTY!r}, "
                    f"got {self.ccp_alpha!r}"
                )
            if not self.honesty:
                # The cross-validation scores subtrees with the honest objective, which
                # is defined by the structure/estimation split. Raise rather than fall
                # back silently, so the setting cannot look applied when it is not.
                raise ValueError(
                    f"ccp_alpha={CV_PENALTY!r} requires honesty=True; it scores "
                    "candidate subtrees with the honest objective, which needs the "
                    "structure/estimation split."
                )

        # Keep original 1d outcomes for post-fit computations
        y_orig = y.copy()

        # The honest criterion of Athey and Imbens (2016) scales its variance penalty
        # by (1 + N^tr / N^est). Read by the builder in ``_tree.py``; 0.0 leaves the
        # penalty unscaled. The override lets the cross-validation fold trees grow
        # with the parent's splitting objective without recomputing it from their own
        # (honesty=False) settings.
        self._train_to_est_ratio = getattr(
            self, "_train_to_est_ratio_override", self._honest_penalty_ratio()
        )
        # Resolved penalty. Stays a float even when ``ccp_alpha`` is the "cv" sentinel,
        # which ``_fit_honest`` replaces with the cross-validated value.
        self.ccp_alpha_ = 0.0 if self.ccp_alpha == CV_PENALTY else self.ccp_alpha

        if prepare_data:
            X, y = self._prepare_data(X=X, y=y, treatment=treatment)

        if self.honesty:
            self._fit_honest(
                X=X,
                treatment=treatment,
                y=y,
                sample_weight=sample_weight,
                check_input=check_input,
            )
        else:
            super().fit(X=X, y=y, sample_weight=sample_weight, check_input=check_input)

        if self.groups_cnt:
            self._groups_cnt = self._count_groups_distribution(X=X, treatment=treatment)
        if self.node_pvalues:
            self._node_pvalues = self._compute_node_pvalues(
                X=X, treatment=treatment, y=y_orig
            )
        return self

    def predict(
        self, X: np.ndarray, with_outcomes: bool = False, check_input=True
    ) -> np.ndarray:
        """Predict individual treatment effects

        Args:
            X (np.ndarray): a feature matrix
            with_outcomes (bool), default=False,
                                  include outcomes Y_hat(X|T=0), Y_hat(X|T=1),...,Y_hat(X|T=n)
                                  along with individual treatment effects
            check_input (bool), default=True,
                                Allow to bypass several input checking.
        Returns:
           (np.ndarray): individual treatment effect (ITE), dim=(samples, groups)
                        or ITE with outcomes:
                        [Y_hat(X|T=0), Y_hat(X|T=1),...,Y_hat(X|T=n), ITE_1, ITE_2,...,ITE_n], dim=(samples, 2*groups-1)
        """
        if check_input:
            X = self._validate_X_predict(X, check_input)
        y_outcomes = super().predict(X)
        y_pred = y_outcomes[:, 1:] - y_outcomes[:, [0]]
        need_outcomes = with_outcomes or self._with_outcomes
        out = np.hstack([y_outcomes, y_pred]) if need_outcomes else y_pred
        # Provides scikit-learn support for _accumulate_prediction() required for causal forests
        if out.shape[1] == 1:
            out = out.ravel()
        return out

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
            X (np.ndarray): a feature matrix
            treatment (np.ndarray): a treatment vector
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
        self.fit(X=X, y=y, treatment=treatment)
        te = self.predict(X=X)

        if return_ci:
            te_bootstraps = self.bootstrap_pool(
                X=X,
                y=y,
                treatment=treatment,
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
            X (np.ndarray): a feature matrix
            treatment (np.array): a treatment vector
            y (np.ndarray): an outcome vector
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
                X=X, y=y, treatment=treatment, sample_size=bootstrap_size, seed=i
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

    def _fit_honest(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: Union[np.ndarray, None],
        check_input: bool,
    ) -> None:
        """Grow the tree on one half of the sample, re-estimate leaves on the other.

        The rows that choose the splits are disjoint from the rows that estimate the
        leaf values, so a leaf value no longer inherits the selection bias of the
        split search that produced it (Athey and Imbens 2016). The split is
        stratified on ``treatment`` so every group is represented in both halves,
        falling back to an unstratified split when a group is too small for that --
        the same fallback ``_KernelUpliftTreeClassifier.fit`` uses.

        Args:
            X: (np.ndarray), feature matrix
            treatment: (np.ndarray), treatment vector, only used to stratify the split
            y: (np.ndarray), outcomes as (samples x groups), from ``_prepare_data``
            sample_weight: (np.ndarray or None), split alongside the rows
            check_input: (bool), forwarded to the structure-split fit
        """
        arrays = [X, y, np.asarray(treatment)]
        if sample_weight is not None:
            arrays.append(sample_weight)
        split_kwargs = dict(
            test_size=self.estimation_sample_size,
            shuffle=True,
            random_state=self.random_state,
        )
        try:
            split = train_test_split(
                *arrays, stratify=np.asarray(treatment), **split_kwargs
            )
        except ValueError:
            split = train_test_split(*arrays, **split_kwargs)

        X_structure, X_estimation, y_structure, y_estimation = split[:4]
        treatment_structure = split[4]
        weight_structure, weight_estimation = (
            split[6:8] if sample_weight is not None else (None, None)
        )

        if self.ccp_alpha == CV_PENALTY:
            self.ccp_alpha_ = self._select_ccp_alpha(
                X=X_structure,
                y=y_structure,
                treatment=treatment_structure,
                sample_weight=weight_structure,
                check_input=check_input,
            )

        super().fit(
            X=X_structure,
            y=y_structure,
            sample_weight=weight_structure,
            check_input=check_input,
        )
        self._honest_reestimate(
            X=X_estimation, y=y_estimation, sample_weight=weight_estimation
        )

    def _honest_penalty_ratio(self) -> float:
        """``N^tr / N^est``, the honest variance-penalty scale, or 0.0 when unused."""
        if not (self.honesty and self.ccp_alpha == CV_PENALTY):
            return 0.0
        return (1.0 - self.estimation_sample_size) / self.estimation_sample_size

    def _honest_objective(self, tree, X: np.ndarray, y: np.ndarray) -> float:
        """``-EMSE_tau(S, Pi)``: the splitting objective evaluated on held-out rows.

        Athey and Imbens (2016) select the cost-complexity penalty by scoring each
        candidate subtree with the *same* objective the splitter maximises, computed
        on a cross-validation sample rather than on the rows that chose the splits::

            sum_leaves  (n_leaf / N) * [ tau_hat^2
                                         - (1 + ratio) * (Var_t / n_t + Var_c / n_c) ]

        A leaf that the held-out fold cannot support -- fewer than two rows in any
        group, so its variance is undefined -- contributes nothing. That is what
        penalises a tree grown deeper than the data can estimate: the extra leaves
        stop earning their ``tau_hat^2``.

        Args:
            tree: a fitted ``Tree`` (pruned or not) to route ``X`` through
            X: (np.ndarray), held-out feature matrix
            y: (np.ndarray), held-out outcomes as (samples x groups)

        Returns:
            (float): the objective; larger is better, and 0.0 if nothing scored.
        """
        leaves = tree.apply(np.ascontiguousarray(X, dtype=DTYPE))
        n_nodes = tree.node_count
        scale = 1.0 + self._train_to_est_ratio

        count, mean, var = [], [], []
        for group in range(y.shape[1]):
            column = y[:, group]
            observed = ~np.isnan(column)
            values = column[observed]
            at = leaves[observed]
            n = np.bincount(at, minlength=n_nodes).astype(np.float64)
            total = np.bincount(at, weights=values, minlength=n_nodes)
            total_sq = np.bincount(at, weights=values**2, minlength=n_nodes)
            safe = np.where(n > 0, n, 1.0)
            group_mean = total / safe
            count.append(n)
            mean.append(group_mean)
            var.append(np.maximum(total_sq / safe - group_mean**2, 0.0))

        is_leaf = tree.children_left == -1
        # Every group needs at least two rows for a usable variance.
        usable = is_leaf & np.all(np.vstack(count) >= 2, axis=0)
        if not usable.any():
            return 0.0

        n_leaf = np.sum(np.vstack(count), axis=0)
        per_group = np.zeros(n_nodes, dtype=np.float64)
        for group in range(1, y.shape[1]):
            tau = mean[group] - mean[0]
            penalty = var[group] / np.where(count[group] > 0, count[group], 1.0) + var[
                0
            ] / np.where(count[0] > 0, count[0], 1.0)
            per_group += tau**2 - scale * penalty
        per_group /= max(y.shape[1] - 1, 1)

        return float(np.sum(n_leaf[usable] * per_group[usable]) / n_leaf.sum())

    def _select_ccp_alpha(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        sample_weight: Union[np.ndarray, None],
        check_input: bool,
    ) -> float:
        """Choose the cost-complexity penalty by cross-validation (Athey-Imbens 2016).

        Each fold grows its own tree on the fold's training rows -- with the parent's
        splitting objective, via ``_train_to_est_ratio_override`` -- and every subtree
        on that tree's cost-complexity path is scored by :meth:`_honest_objective` on
        the held-out rows. The penalty with the best average score wins. Pruning
        reuses the already-grown fold tree rather than refitting per candidate, so the
        cost is ``cv_folds`` fits, not ``cv_folds x n_alphas``.

        Returns:
            (float): the selected ``ccp_alpha``; 0.0 (no pruning) if CV is not usable.
        """
        splitter = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        try:
            folds = list(splitter.split(X, treatment))
        except ValueError:
            return 0.0

        totals, counts = {}, {}
        for train_idx, test_idx in folds:
            fold_tree = self._make_fold_tree()
            try:
                fold_tree.fit(
                    X=X[train_idx],
                    treatment=treatment[train_idx],
                    y=y[train_idx],
                    sample_weight=(
                        None if sample_weight is None else sample_weight[train_idx]
                    ),
                    check_input=check_input,
                    prepare_data=False,
                )
            except (ValueError, IndexError):
                continue

            alphas = np.unique(ccp_pruning_path(fold_tree.tree_)["ccp_alphas"])
            alphas = alphas[alphas >= 0.0]
            if alphas.size > MAX_CCP_CANDIDATES:
                keep = np.linspace(0, alphas.size - 1, MAX_CCP_CANDIDATES).astype(int)
                alphas = alphas[keep]

            for alpha in alphas:
                pruned = Tree(
                    fold_tree.n_features_in_,
                    np.array([1] * fold_tree.n_outputs_, dtype=np.intp),
                    fold_tree.n_outputs_,
                )
                _build_pruned_tree_ccp(pruned, fold_tree.tree_, float(alpha))
                score = self._honest_objective(pruned, X[test_idx], y[test_idx])
                key = float(alpha)
                totals[key] = totals.get(key, 0.0) + score
                counts[key] = counts.get(key, 0) + 1

        if not totals:
            return 0.0
        # Average over the folds that offered each candidate, then break ties toward
        # the larger penalty: among equally-scoring trees the paper prefers the
        # smaller one.
        best = max(sorted(totals), key=lambda a: totals[a] / counts[a])
        return float(best)

    def _make_fold_tree(self) -> "CausalTreeRegressor":
        """An adaptive clone used to grow one cross-validation fold.

        Honesty and the cross-validation are both off -- the fold tree is grown on
        its fold's rows and scored on the held-out ones, so it neither re-splits nor
        recurses -- but it keeps the parent's splitting objective so the candidate
        subtrees are the ones the final tree would produce.
        """
        params = self.get_params()
        params.update(honesty=False, ccp_alpha=0.0)
        fold_tree = CausalTreeRegressor(**params)
        fold_tree._train_to_est_ratio_override = self._train_to_est_ratio
        return fold_tree

    def _honest_reestimate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Union[np.ndarray, None],
    ) -> None:
        """Overwrite each leaf's per-group outcome mean from the estimation split.

        A causal-tree leaf holds one mean outcome per group (control first) and
        ``predict`` differences them into the treatment effect, so re-estimating
        those means is what makes the effect honest. ``tree_.value`` is a writable
        view on the tree's own buffer, so the assignment reaches the values that
        predictions read.

        A group with no estimation rows in a leaf keeps its structure-split value:
        ``min_group_samples`` is enforced by the builder on the structure half only,
        and writing a zero (or a NaN) for the missing group would corrupt that leaf's
        effect rather than leave it merely stale.

        Args:
            X: (np.ndarray), estimation-split feature matrix
            y: (np.ndarray), estimation-split outcomes as (samples x groups)
            sample_weight: (np.ndarray or None), weights for the estimation rows
        """
        X = self._validate_X_predict(X, check_input=True)
        leaf_ids = self.tree_.apply(X)
        value = self.tree_.value  # (node_count, n_groups, 1), writable view
        n_nodes = self.tree_.node_count
        is_leaf = self.tree_.children_left == -1

        for group in range(self.n_outputs_):
            outcomes = y[:, group]
            observed = ~np.isnan(outcomes)
            leaves = leaf_ids[observed]
            if sample_weight is None:
                weights = np.ones(leaves.shape[0], dtype=np.float64)
            else:
                weights = np.asarray(sample_weight, dtype=np.float64)[observed]
            total_weight = np.bincount(leaves, weights=weights, minlength=n_nodes)
            total_outcome = np.bincount(
                leaves, weights=weights * outcomes[observed], minlength=n_nodes
            )
            estimated = is_leaf & (total_weight > 0)
            value[estimated, group, 0] = (
                total_outcome[estimated] / total_weight[estimated]
            )

    def _prepare_data(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare input data with treatment info for DecisionTreeRegressor.
        Outcome vector y transforms into y_2dim with (samples x groups) dimensions.
        Outcomes for the control group are always placed in the first column with index 0.
        Attribute _group2index stores mapping for y_2dim columns: ({control: 0, treatmentA: 1, treatmentB: 2, ...})
        Args:
            X: : (np.ndarray), feature matrix
            treatment: : (np.ndarray), treatment vector, includes control group
            y: : (np.ndarray), outcome vector
        Returns: X, y (samples x groups)
        """
        if y.shape[0] != treatment.shape[0]:
            raise ValueError(
                f"The number of `treatment` and `y` rows are not equal: {y.shape[0]} {treatment.shape[0]}"
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

        # Missing (NaN) feature values are handled natively by the kernel
        # splitter; finiteness is enforced (support-gated) by the base fit's
        # ``_compute_missing_values_in_feature_mask``.
        X = check_array(
            X, dtype=DTYPE, accept_sparse="csc", ensure_all_finite="allow-nan"
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        self.n_samples, self.n_features = X.shape

        y_2dim = np.zeros((self.n_samples, len(self.unique_treatments) + 1))
        for group, group_index in self._group2index.items():
            y_2dim[:, group_index] = np.where(treatment == group, y, np.nan)

        return X, y_2dim

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
        groups = np.unique(treatment)
        groups_cnt = {
            idx: {group: 0 for group in groups}
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

    def _compute_node_pvalues(
        self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray
    ) -> dict:
        """
        Compute treatment effect p-values for each tree node using Welch's t-test.

        Note: These p-values are descriptive and do not account for the search
        process used to find the splits (no honesty or multiple-testing correction).

        Args:
            X: (np.ndarray), feature matrix
            treatment: (np.ndarray), treatment vector
            y: (np.ndarray), outcome vector (1d, original outcomes)

        Returns:
            dict: {node_id: {treatment_group: p_value}} for each node
        """
        check_is_fitted(self)

        groups = sorted(set(treatment))
        control = self.control_name
        treatments = [g for g in groups if g != control]

        # Collect outcomes per group per node
        node_outcomes = {
            idx: {group: [] for group in groups} for idx in range(self.tree_.node_count)
        }

        node_indicators = self.tree_.decision_path(X.astype(np.float32))
        for sample_id in range(X.shape[0]):
            nodes_path = node_indicators.indices[
                node_indicators.indptr[sample_id] : node_indicators.indptr[
                    sample_id + 1
                ]
            ]
            group = treatment[sample_id]
            outcome = y[sample_id]
            for node_id in nodes_path:
                node_outcomes[node_id][group].append(outcome)

        # Compute p-values via Welch's t-test (treatment vs control)
        node_pvalues = {}
        for node_id in range(self.tree_.node_count):
            control_outcomes = node_outcomes[node_id][control]
            pvals = {}
            for t in treatments:
                treatment_outcomes = node_outcomes[node_id][t]
                if len(control_outcomes) >= 2 and len(treatment_outcomes) >= 2:
                    _, p = ttest_ind(
                        treatment_outcomes, control_outcomes, equal_var=False
                    )
                    pvals[t] = round(float(p), 4)
                else:
                    pvals[t] = None
            node_pvalues[node_id] = pvals

        return node_pvalues
