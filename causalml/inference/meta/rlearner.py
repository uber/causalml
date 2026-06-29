from copy import deepcopy
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from xgboost import XGBRegressor

from causalml.inference.meta.base import BaseLearner
from causalml.inference.meta.utils import (
    check_treatment_vector,
    collect_if_lazy,
    filter_mask,
    n_rows,
    to_numpy,
    get_xgboost_objective_metric,
    get_weighted_variance,
    convert_pd_to_np,
)
from causalml.propensity import ElasticNetPropensityModel

logger = logging.getLogger("causalml")


class BaseRLearner(BaseLearner):
    """A parent class for R-learner classes.

    An R-learner estimates treatment effects with two machine learning models and the propensity score.

    Details of R-learner are available at `Nie and Wager (2019) <https://arxiv.org/abs/1712.04912>`_.
    """

    def __init__(
        self,
        learner=None,
        outcome_learner=None,
        effect_learner=None,
        propensity_learner=ElasticNetPropensityModel(),
        ate_alpha=0.05,
        control_name=0,
        n_fold=5,
        random_state=None,
        cv_n_jobs=-1,
    ):
        """Initialize an R-learner.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects
            outcome_learner (optional): a model to estimate outcomes
            effect_learner (optional): a model to estimate treatment effects. It needs to take `sample_weight` as an
                input argument for `fit()`
            propensity_learner (optional): a model to estimate propensity scores. `ElasticNetPropensityModel()` will
                be used by default.
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generator (RandomState)
            cv_n_jobs (int, optional): number of parallel jobs to run for cross_val_predict. -1 means using all
                processors

        Note: arguments are stored verbatim (scikit-learn convention) so that
        ``get_params`` / ``clone`` work correctly. Model construction is deferred to ``fit()``.
        Per the scikit-learn convention, ``__init__`` does not validate or raise —
        validation of ``learner``/``outcome_learner``/``effect_learner`` happens in ``fit()``.
        """
        # Store verbatim — no deepcopy, no logic (scikit-learn convention).
        self.learner = learner
        self.outcome_learner = outcome_learner
        self.effect_learner = effect_learner
        self.propensity_learner = propensity_learner
        self.ate_alpha = ate_alpha
        self.control_name = control_name
        self.n_fold = n_fold
        self.random_state = random_state
        self.cv_n_jobs = cv_n_jobs

    def fit(self, X, treatment, y, p=None, sample_weight=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method; the
                feature matrix is otherwise kept in its native format throughout,
                including the call to ``cross_val_predict`` (scikit-learn >= 1.6
                accepts pandas and Polars DataFrames natively).
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array, pd.Series, or pl.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            verbose (bool, optional): whether to output progress logs

            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): propensity scores
            sample_weight (np.array or pd.Series, optional): sample weights for `effect_learner`.
            verbose (bool, optional): whether to output progress logs
        """
        X = collect_if_lazy(X)
        if (self.learner is None) and (
            (self.outcome_learner is None) or (self.effect_learner is None)
        ):
            raise ValueError(
                "Either `learner` or both `outcome_learner` and `effect_learner` "
                "must be specified."
            )
        if self.propensity_learner is None:
            raise ValueError("`propensity_learner` must be specified.")
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        if sample_weight is not None:
            assert len(sample_weight) == len(
                y_np
            ), "Data length must be equal for sample_weight and the input data"
            sample_weight = to_numpy(sample_weight)

        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment_np, y=y_np)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        # Resolve base models from stored constructor args (scikit-learn convention).
        self.model_mu = (
            self.outcome_learner
            if self.outcome_learner is not None
            else deepcopy(self.learner)
        )
        self.model_tau = (
            self.effect_learner
            if self.effect_learner is not None
            else deepcopy(self.learner)
        )
        self.model_p = self.propensity_learner
        # Build CV splitter from stored n_fold / random_state.
        self.cv = KFold(
            n_splits=self.n_fold, shuffle=True, random_state=self.random_state
        )

        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(
            self.model_mu, X, y_np, cv=self.cv, n_jobs=self.cv_n_jobs
        )

        for group in self.t_groups:
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = filter_mask(treatment, mask)
            X_filt = filter_mask(X, mask)
            y_filt = y_np[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            w = (to_numpy(treatment_filt) == group).astype(int)

            weight = (w - p_filt) ** 2
            diff_c = y_filt[w == 0] - yhat_filt[w == 0]
            diff_t = y_filt[w == 1] - yhat_filt[w == 1]
            if sample_weight is not None:
                sample_weight_filt = sample_weight[mask]

                sample_weight_filt_c = sample_weight_filt[w == 0]
                sample_weight_filt_t = sample_weight_filt[w == 1]
                self.vars_c[group] = get_weighted_variance(diff_c, sample_weight_filt_c)
                self.vars_t[group] = get_weighted_variance(diff_t, sample_weight_filt_t)
                weight *= sample_weight_filt
            else:
                self.vars_c[group] = diff_c.var()
                self.vars_t[group] = diff_t.var()

            if verbose:
                logger.info(
                    "training the treatment effect model for {} with R-loss".format(
                        group
                    )
                )
            self.models_tau[group].fit(
                X_filt, (y_filt - yhat_filt) / (w - p_filt), sample_weight=weight
            )
        return self

    def predict(self, X, p=None):
        """Predict treatment effects.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method.

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = collect_if_lazy(X)
        te = np.zeros((n_rows(X), self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = self.models_tau[group].predict(X)
        return te

    def fit_predict(
        self,
        X,
        treatment,
        y,
        p=None,
        sample_weight=None,
        return_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        verbose=True,
    ):
        """Fit the R learner and predict treatment effects.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array, pd.Series, or pl.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): propensity scores
            sample_weight (np.array or pd.Series, optional): sample weights
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            verbose (bool): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = collect_if_lazy(X)
        self.fit(X, treatment, y, p, sample_weight, verbose=verbose)
        te = self.predict(X)

        if not return_ci:
            return te
        else:
            treatment_np = to_numpy(treatment)
            y_np = to_numpy(y)

            t_groups_global = self.t_groups
            _classes_global = self._classes
            model_mu_global = deepcopy(self.model_mu)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(
                shape=(n_rows(X), self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                if p is None:
                    p = self.propensity
                else:
                    p = self._format_p(p, self.t_groups)
                te_b = self.bootstrap(X, treatment_np, y_np, p, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(
                te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2
            )

            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.model_mu = deepcopy(model_mu_global)
            self.models_tau = deepcopy(models_tau_global)

            return (te, te_lower, te_upper)

    def estimate_ate(
        self,
        X,
        treatment=None,
        y=None,
        p=None,
        sample_weight=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        pretrain=False,
    ):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix
            treatment (np.array, pd.Series, or pl.Series): only needed when pretrain=False, a treatment vector
            y (np.array, pd.Series, or pl.Series): only needed when pretrain=False, an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array, pd.Series, or pl.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): treatment vector (needed when pretrain=False)
            y (np.array or pd.Series): outcome vector (needed when pretrain=False)
            p (np.ndarray or pd.Series or dict, optional): propensity scores
            sample_weight (np.array or pd.Series, optional): sample weights
            bootstrap_ci (bool): whether run bootstrap for confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            pretrain (bool): whether a model has been fit, default False.
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        X = collect_if_lazy(X)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        if pretrain:
            te = self.predict(X, p)
        else:
            if treatment is None or y is None:
                raise ValueError("treatment and y must be provided when pretrain=False")

            te = self.fit_predict(
                X,
                treatment,
                y,
                p,
                sample_weight,
                return_ci=False,
            )

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            w = (treatment_np == group).astype(int)
            prob_treatment = float(sum(w)) / n_rows(X)
            _ate = te[:, i].mean()

            se = np.sqrt(
                (self.vars_t[group] / prob_treatment)
                + (self.vars_c[group] / (1 - prob_treatment))
                + te[:, i].var()
            ) / n_rows(X)

            _ate_lb = _ate - se * norm.ppf(1 - self.ate_alpha / 2)
            _ate_ub = _ate + se * norm.ppf(1 - self.ate_alpha / 2)

            ate[i] = _ate
            ate_lb[i] = _ate_lb
            ate_ub[i] = _ate_ub

        if not bootstrap_ci:
            return ate, ate_lb, ate_ub
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            model_mu_global = deepcopy(self.model_mu)
            models_tau_global = deepcopy(self.models_tau)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                if p is None:
                    p = self.propensity
                else:
                    p = self._format_p(p, self.t_groups)
                cate_b = self.bootstrap(X, treatment_np, y_np, p, size=bootstrap_size)
                ate_bootstraps[:, n] = cate_b.mean(axis=0)

            ate_lower = np.percentile(
                ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1
            )
            ate_upper = np.percentile(
                ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1
            )

            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.model_mu = deepcopy(model_mu_global)
            self.models_tau = deepcopy(models_tau_global)
            return ate, ate_lower, ate_upper


class BaseRRegressor(BaseRLearner):
    """A parent class for R-learner regressor classes."""

    def __init__(
        self,
        learner=None,
        outcome_learner=None,
        effect_learner=None,
        propensity_learner=ElasticNetPropensityModel(),
        ate_alpha=0.05,
        control_name=0,
        n_fold=5,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            propensity_learner=propensity_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
            n_fold=n_fold,
            random_state=random_state,
        )


class BaseRClassifier(BaseRLearner):
    """A parent class for R-learner classifier classes."""

    def __init__(
        self,
        outcome_learner=None,
        effect_learner=None,
        propensity_learner=ElasticNetPropensityModel(),
        ate_alpha=0.05,
        control_name=0,
        n_fold=5,
        random_state=None,
    ):
        """Initialize an R-learner classifier.

        Args:
            outcome_learner: a classifier for outcomes.
            effect_learner: a regressor for treatment effects (needs ``sample_weight`` in fit).
            propensity_learner (optional): a propensity model. Defaults to ElasticNetPropensityModel.
            ate_alpha (float, optional): confidence level alpha
            control_name (str or int, optional): name of control group
            n_fold (int, optional): CV folds for outcome_learner
            random_state (int or RandomState, optional): random seed
        """
        if (outcome_learner is None) and (effect_learner is None):
            raise ValueError(
                "Either the outcome learner or the effect learner must be specified."
            )

        super().__init__(
            learner=None,
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            propensity_learner=propensity_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
            n_fold=n_fold,
            random_state=random_state,
        )

    def fit(self, X, treatment, y, p=None, sample_weight=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method.
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array, pd.Series, or pl.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            verbose (bool, optional): whether to output progress logs
        """
        X = collect_if_lazy(X)
        """Fit the R-learner classifier (uses predict_proba for outcome estimates)."""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        if sample_weight is not None:
            assert len(sample_weight) == len(
                y_np
            ), "Data length must be equal for sample_weight and the input data"
            sample_weight = to_numpy(sample_weight)

        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment_np, y=y_np)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        # Resolve base models from stored constructor args.
        self.model_mu = self.outcome_learner
        self.model_tau = self.effect_learner
        self.model_p = self.propensity_learner
        self.cv = KFold(
            n_splits=self.n_fold, shuffle=True, random_state=self.random_state
        )

        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(
            self.model_mu, X, y_np, cv=self.cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        for group in self.t_groups:
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = filter_mask(treatment, mask)
            X_filt = filter_mask(X, mask)
            y_filt = y_np[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            w = (to_numpy(treatment_filt) == group).astype(int)

            weight = (w - p_filt) ** 2
            diff_c = y_filt[w == 0] - yhat_filt[w == 0]
            diff_t = y_filt[w == 1] - yhat_filt[w == 1]
            if sample_weight is not None:
                sample_weight_filt = sample_weight[mask]
                self.vars_c[group] = get_weighted_variance(
                    diff_c, sample_weight_filt[w == 0]
                )
                self.vars_t[group] = get_weighted_variance(
                    diff_t, sample_weight_filt[w == 1]
                )
                sample_weight_filt_c = sample_weight_filt[w == 0]
                sample_weight_filt_t = sample_weight_filt[w == 1]
                self.vars_c[group] = get_weighted_variance(diff_c, sample_weight_filt_c)
                self.vars_t[group] = get_weighted_variance(diff_t, sample_weight_filt_t)
                weight *= sample_weight_filt
            else:
                self.vars_c[group] = diff_c.var()
                self.vars_t[group] = diff_t.var()

            if verbose:
                logger.info(
                    "training the treatment effect model for {} with R-loss".format(
                        group
                    )
                )
            self.models_tau[group].fit(
                X_filt, (y_filt - yhat_filt) / (w - p_filt), sample_weight=weight
            )
        return self

    def predict(self, X, p=None):
        """Predict treatment effects.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method.

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = collect_if_lazy(X)
        te = np.zeros((n_rows(X), self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = self.models_tau[group].predict(X)
        return te


class XGBRRegressor(BaseRRegressor):
    """An R-learner regressor using XGBoost models.

    Stores every constructor argument verbatim (scikit-learn convention) so
    that ``get_params()`` / ``clone()`` work correctly. All XGBRegressor
    construction is deferred to ``fit()``.

    Additional XGBoost keyword arguments (e.g. ``max_depth``, ``learning_rate``)
    are accepted via ``**xgb_kwargs`` and stored verbatim as ``self.xgb_kwargs``,
    so that ``get_params()`` surfaces them and ``clone()`` round-trips them
    correctly.
    """

    def __init__(
        self,
        early_stopping=True,
        test_size=0.3,
        early_stopping_rounds=30,
        effect_learner_objective="reg:squarederror",
        effect_learner_n_estimators=500,
        random_state=42,
        ate_alpha=0.05,
        control_name=0,
        n_fold=5,
        xgb_kwargs=None,
    ):
        """Initialize an R-learner regressor with XGBoost models.

        Args:
            early_stopping (bool, optional): whether to use early stopping for the effect learner
            test_size (float, optional): held-out fraction for early stopping eval set
            early_stopping_rounds (int, optional): early stopping patience
            effect_learner_objective (str, optional): XGBoost objective for the effect learner
            effect_learner_n_estimators (int, optional): n_estimators for the effect learner
            random_state (int, optional): random seed (must be int)
            ate_alpha (float, optional): confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): CV folds for the outcome learner
            xgb_kwargs (dict, optional): additional keyword arguments forwarded verbatim
                to both XGBRegressor instances (outcome and effect learners), e.g.
                ``xgb_kwargs={'max_depth': 4, 'learning_rate': 0.05}``.

        Note: all arguments are stored verbatim (scikit-learn convention) so that
        ``get_params`` / ``clone`` work correctly. XGBRegressor construction is
        deferred to ``fit()``.
        """
        assert isinstance(random_state, int), "random_state should be int."

        # Store verbatim — no transformation, no XGBRegressor construction here.
        # xgb_kwargs=None is stored as-is; BaseEstimator.get_params surfaces it
        # correctly since it is a named parameter.  The or {} coalesce happens in
        # fit() so that clone(XGBRRegressor()) still round-trips None → None.
        self.early_stopping = early_stopping
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds
        self.effect_learner_objective = effect_learner_objective
        self.effect_learner_n_estimators = effect_learner_n_estimators
        self.xgb_kwargs = xgb_kwargs

        super().__init__(
            learner=None,
            outcome_learner=None,
            effect_learner=None,
            ate_alpha=ate_alpha,
            control_name=control_name,
            n_fold=n_fold,
            random_state=random_state,
        )

    def fit(self, X, treatment, y, p=None, sample_weight=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method.
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array, pd.Series, or pl.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            verbose (bool, optional): whether to output progress logs
        """
        X = collect_if_lazy(X)
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        # initialize equal sample weight if it's not provided, for simplicity purpose
        """Fit using early-stopping XGBoost R-learner."""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        sample_weight = (
            to_numpy(sample_weight) if sample_weight is not None else np.ones(len(y_np))
        )
        assert len(sample_weight) == len(
            y_np
        ), "Data length must be equal for sample_weight and the input data"
        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment_np, y=y_np)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        # Resolve XGBRegressor models here (not in __init__) so get_params/clone
        # stay correct — the constructor only stores plain, verbatim values.
        # self.xgb_kwargs holds any extra XGBoost params (e.g. max_depth) verbatim.
        objective, metric = get_xgboost_objective_metric(self.effect_learner_objective)
        xgb_kw = self.xgb_kwargs or {}
        if self.early_stopping:
            effect_learner = XGBRegressor(
                objective=objective,
                n_estimators=self.effect_learner_n_estimators,
                eval_metric=metric,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=self.random_state,
                **xgb_kw,
            )
        else:
            effect_learner = XGBRegressor(
                objective=objective,
                n_estimators=self.effect_learner_n_estimators,
                eval_metric=metric,
                random_state=self.random_state,
                **xgb_kw,
            )
        outcome_learner = XGBRegressor(random_state=self.random_state, **xgb_kw)

        self.model_mu = outcome_learner
        self.model_tau = effect_learner
        self.model_p = self.propensity_learner
        self.cv = KFold(
            n_splits=self.n_fold, shuffle=True, random_state=self.random_state
        )

        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(self.model_mu, X, y_np, cv=self.cv, n_jobs=-1)

        for group in self.t_groups:
            treatment_mask = (treatment_np == group) | (
                treatment_np == self.control_name
            )
            treatment_filt = filter_mask(treatment, treatment_mask)
            w = (to_numpy(treatment_filt) == group).astype(int)

            X_filt = filter_mask(X, treatment_mask)
            y_filt = y_np[treatment_mask]
            yhat_filt = yhat[treatment_mask]
            p_filt = p[group][treatment_mask]
            sample_weight_filt = sample_weight[treatment_mask]

            if verbose:
                logger.info(
                    "training the treatment effect model for {} with R-loss".format(
                        group
                    )
                )

            if self.early_stopping:
                (
                    X_train_filt,
                    X_test_filt,
                    y_train_filt,
                    y_test_filt,
                    yhat_train_filt,
                    yhat_test_filt,
                    w_train,
                    w_test,
                    p_train_filt,
                    p_test_filt,
                    sample_weight_train_filt,
                    sample_weight_test_filt,
                ) = train_test_split(
                    X_filt,
                    y_filt,
                    yhat_filt,
                    w,
                    p_filt,
                    sample_weight_filt,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

                self.models_tau[group].fit(
                    X=X_train_filt,
                    y=(y_train_filt - yhat_train_filt) / (w_train - p_train_filt),
                    sample_weight=sample_weight_train_filt
                    * ((w_train - p_train_filt) ** 2),
                    eval_set=[
                        (
                            X_test_filt,
                            (y_test_filt - yhat_test_filt) / (w_test - p_test_filt),
                        )
                    ],
                    sample_weight_eval_set=[
                        sample_weight_test_filt * ((w_test - p_test_filt) ** 2)
                    ],
                    verbose=verbose,
                )
            else:
                self.models_tau[group].fit(
                    X_filt,
                    (y_filt - yhat_filt) / (w - p_filt),
                    sample_weight=sample_weight_filt * ((w - p_filt) ** 2),
                )

            diff_c = y_filt[w == 0] - yhat_filt[w == 0]
            diff_t = y_filt[w == 1] - yhat_filt[w == 1]
            sample_weight_filt_c = sample_weight_filt[w == 0]
            sample_weight_filt_t = sample_weight_filt[w == 1]
            self.vars_c[group] = get_weighted_variance(diff_c, sample_weight_filt_c)
            self.vars_t[group] = get_weighted_variance(diff_t, sample_weight_filt_t)
        return self
