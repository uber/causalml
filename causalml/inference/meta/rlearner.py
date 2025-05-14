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
    get_xgboost_objective_metric,
    convert_pd_to_np,
    get_weighted_variance,
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
        """
        assert (learner is not None) or (
            (outcome_learner is not None) and (effect_learner is not None)
        )
        assert propensity_learner is not None

        self.model_mu = (
            outcome_learner if outcome_learner is not None else deepcopy(learner)
        )
        self.model_tau = (
            effect_learner if effect_learner is not None else deepcopy(learner)
        )
        self.model_p = propensity_learner

        self.ate_alpha = ate_alpha
        self.control_name = control_name

        self.random_state = random_state
        self.cv = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
        self.cv_n_jobs = cv_n_jobs

        self.propensity = None
        self.propensity_model = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"\toutcome_learner={self.model_mu.__repr__()}\n"
            f"\teffect_learner={self.model_tau.__repr__()}\n"
            f"\tpropensity_learner={self.model_p.__repr__()}"
        )

    def fit(self, X, treatment, y, p=None, sample_weight=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array or pd.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            verbose (bool, optional): whether to output progress logs
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        if sample_weight is not None:
            assert len(sample_weight) == len(
                y
            ), "Data length must be equal for sample_weight and the input data"
            sample_weight = convert_pd_to_np(sample_weight)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment, y=y)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv, n_jobs=self.cv_n_jobs)

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            w = (treatment_filt == group).astype(int)

            weight = (w - p_filt) ** 2
            diff_c = y_filt[w == 0] - yhat_filt[w == 0]
            diff_t = y_filt[w == 1] - yhat_filt[w == 1]
            if sample_weight is not None:
                sample_weight_filt = sample_weight[mask]
                sample_weight_filt_c = sample_weight_filt[w == 0]
                sample_weight_filt_t = sample_weight_filt[w == 1]
                self.vars_c[group] = get_weighted_variance(diff_c, sample_weight_filt_c)
                self.vars_t[group] = get_weighted_variance(diff_t, sample_weight_filt_t)
                weight *= sample_weight_filt  # update weight
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

    def predict(self, X, p=None):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = convert_pd_to_np(X)
        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            dhat = self.models_tau[group].predict(X)
            te[:, i] = dhat

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
        """Fit the treatment effect and outcome models of the R learner and predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array or pd.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            verbose (bool): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment].
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        self.fit(X, treatment, y, p, sample_weight, verbose=verbose)
        te = self.predict(X)

        if not return_ci:
            return te
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            model_mu_global = deepcopy(self.model_mu)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(
                shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                if p is None:
                    p = self.propensity
                else:
                    p = self._format_p(p, self.t_groups)
                te_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(
                te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2
            )

            # set member variables back to global (currently last bootstrapped outcome)
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
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): only needed when pretrain=False, a treatment vector
            y (np.array or pd.Series):only needed when pretrain=False, an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array or pd.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            bootstrap_ci (bool): whether run bootstrap for confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            pretrain (bool): whether a model has been fit, default False.
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        if pretrain:
            te = self.predict(X, p)
        else:
            if not len(treatment) or not len(y):
                raise ValueError("treatmeng and y must be provided when pretrain=False")
            te = self.fit_predict(X, treatment, y, p, sample_weight, return_ci=False)

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            w = (treatment == group).astype(int)
            prob_treatment = float(sum(w)) / X.shape[0]
            _ate = te[:, i].mean()

            se = (
                np.sqrt(
                    (self.vars_t[group] / prob_treatment)
                    + (self.vars_c[group] / (1 - prob_treatment))
                    + te[:, i].var()
                )
                / X.shape[0]
            )

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
                cate_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size)
                ate_bootstraps[:, n] = cate_b.mean(axis=0)

            ate_lower = np.percentile(
                ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1
            )
            ate_upper = np.percentile(
                ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1
            )

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.model_mu = deepcopy(model_mu_global)
            self.models_tau = deepcopy(models_tau_global)
            return ate, ate_lower, ate_upper


class BaseRRegressor(BaseRLearner):
    """
    A parent class for R-learner regressor classes.
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
    ):
        """Initialize an R-learner regressor.

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
        """
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
    """
    A parent class for R-learner classifier classes.
    """

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
            outcome_learner: a model to estimate outcomes. Should be a classifier.
            effect_learner: a model to estimate treatment effects. It needs to take `sample_weight` as an
                input argument for `fit()`. Should be a regressor.
            propensity_learner (optional): a model to estimate propensity scores. `ElasticNetPropensityModel()` will
                be used by default.
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generator (RandomState)
        """
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

        if (outcome_learner is None) and (effect_learner is None):
            raise ValueError(
                "Either the outcome learner or the effect learner must be specified."
            )

    def fit(self, X, treatment, y, p=None, sample_weight=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array or pd.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            verbose (bool, optional): whether to output progress logs
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        if sample_weight is not None:
            assert len(sample_weight) == len(
                y
            ), "Data length must be equal for sample_weight and the input data"
            sample_weight = convert_pd_to_np(sample_weight)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment, y=y)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(
            self.model_mu, X, y, cv=self.cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            w = (treatment_filt == group).astype(int)

            weight = (w - p_filt) ** 2
            diff_c = y_filt[w == 0] - yhat_filt[w == 0]
            diff_t = y_filt[w == 1] - yhat_filt[w == 1]
            if sample_weight is not None:
                sample_weight_filt = sample_weight[mask]
                sample_weight_filt_c = sample_weight_filt[w == 0]
                sample_weight_filt_t = sample_weight_filt[w == 1]
                self.vars_c[group] = get_weighted_variance(diff_c, sample_weight_filt_c)
                self.vars_t[group] = get_weighted_variance(diff_t, sample_weight_filt_t)
                weight *= sample_weight_filt  # update weight
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

    def predict(self, X, p=None):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = convert_pd_to_np(X)
        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            dhat = self.models_tau[group].predict(X)
            te[:, i] = dhat

        return te


class XGBRRegressor(BaseRRegressor):
    def __init__(
        self,
        early_stopping=True,
        test_size=0.3,
        early_stopping_rounds=30,
        effect_learner_objective="reg:squarederror",
        effect_learner_n_estimators=500,
        random_state=42,
        *args,
        **kwargs,
    ):
        """Initialize an R-learner regressor with XGBoost model using pairwise ranking objective.

        Args:
            early_stopping: whether or not to use early stopping when fitting effect learner
            test_size (float, optional): the proportion of the dataset to use as validation set when early stopping is
                                         enabled
            early_stopping_rounds (int, optional): validation metric needs to improve at least once in every
                                                   early_stopping_rounds round(s) to continue training
            effect_learner_objective (str, optional): the learning objective for the effect learner
                                                      (default = 'reg:squarederror')
            effect_learner_n_estimators (int, optional): number of trees to fit for the effect learner (default = 500)
        """

        assert isinstance(random_state, int), "random_state should be int."

        objective, metric = get_xgboost_objective_metric(effect_learner_objective)
        self.effect_learner_objective = objective
        self.effect_learner_eval_metric = metric
        self.effect_learner_n_estimators = effect_learner_n_estimators
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.test_size = test_size
            self.early_stopping_rounds = early_stopping_rounds

            effect_learner = XGBRegressor(
                objective=self.effect_learner_objective,
                n_estimators=self.effect_learner_n_estimators,
                eval_metric=self.effect_learner_eval_metric,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=random_state,
                *args,
                **kwargs,
            )
        else:
            effect_learner = XGBRegressor(
                objective=self.effect_learner_objective,
                n_estimators=self.effect_learner_n_estimators,
                eval_metric=self.effect_learner_eval_metric,
                random_state=random_state,
                *args,
                **kwargs,
            )

        super().__init__(
            outcome_learner=XGBRegressor(random_state=random_state, *args, **kwargs),
            effect_learner=effect_learner,
        )

    def fit(self, X, treatment, y, p=None, sample_weight=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            sample_weight (np.array or pd.Series, optional): an array of sample weights indicating the
                weight of each observation for `effect_learner`. If None, it assumes equal weight.
            verbose (bool, optional): whether to output progress logs
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        # initialize equal sample weight if it's not provided, for simplicity purpose
        sample_weight = (
            convert_pd_to_np(sample_weight)
            if sample_weight is not None
            else convert_pd_to_np(np.ones(len(y)))
        )
        assert len(sample_weight) == len(
            y
        ), "Data length must be equal for sample_weight and the input data"
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment, y=y)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv, n_jobs=-1)

        for group in self.t_groups:
            treatment_mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[treatment_mask]
            w = (treatment_filt == group).astype(int)

            X_filt = X[treatment_mask]
            y_filt = y[treatment_mask]
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
