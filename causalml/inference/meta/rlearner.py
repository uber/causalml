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
    filter_mask,
    to_numpy,
    get_xgboost_objective_metric,
    get_weighted_variance,
)
from causalml.propensity import ElasticNetPropensityModel

logger = logging.getLogger("causalml")


class BaseRLearner(BaseLearner):
    """A parent class for R-learner classes."""

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
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        if sample_weight is not None:
            assert len(sample_weight) == len(
                y
            ), "Data length must be equal for sample_weight and the input data"
            sample_weight = to_numpy(sample_weight)

        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=to_numpy(X), treatment=treatment_np, y=y_np)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        # sklearn >= 1.6 accepts DataFrames natively — pass X as-is
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
                self.vars_c[group] = get_weighted_variance(
                    diff_c, sample_weight_filt[w == 0]
                )
                self.vars_t[group] = get_weighted_variance(
                    diff_t, sample_weight_filt[w == 1]
                )
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

    def predict(self, X, p=None):
        X_np = to_numpy(X)
        te = np.zeros((X_np.shape[0], self.t_groups.shape[0]))
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
        self.fit(X, treatment, y, p, sample_weight, verbose=verbose)
        te = self.predict(X)

        if not return_ci:
            return te
        else:
            X_np = to_numpy(X)
            treatment_np = to_numpy(treatment)
            y_np = to_numpy(y)

            t_groups_global = self.t_groups
            _classes_global = self._classes
            model_mu_global = deepcopy(self.model_mu)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(
                shape=(X_np.shape[0], self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                if p is None:
                    p = self.propensity
                else:
                    p = self._format_p(p, self.t_groups)
                te_b = self.bootstrap(X_np, treatment_np, y_np, p, size=bootstrap_size)
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
        treatment_np = to_numpy(treatment)
        X_np = to_numpy(X)

        if pretrain:
            te = self.predict(X, p)
        else:
            if not len(treatment_np) or not len(to_numpy(y)):
                raise ValueError("treatment and y must be provided when pretrain=False")
            te = self.fit_predict(X, treatment, y, p, sample_weight, return_ci=False)

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            w = (treatment_np == group).astype(int)
            prob_treatment = float(sum(w)) / X_np.shape[0]
            _ate = te[:, i].mean()

            se = (
                np.sqrt(
                    (self.vars_t[group] / prob_treatment)
                    + (self.vars_c[group] / (1 - prob_treatment))
                    + te[:, i].var()
                )
                / X_np.shape[0]
            )

            _ate_lb = _ate - se * norm.ppf(1 - self.ate_alpha / 2)
            _ate_ub = _ate + se * norm.ppf(1 - self.ate_alpha / 2)

            ate[i] = _ate
            ate_lb[i] = _ate_lb
            ate_ub[i] = _ate_ub

        if not bootstrap_ci:
            return ate, ate_lb, ate_ub
        else:
            y_np = to_numpy(y)
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
                cate_b = self.bootstrap(
                    X_np, treatment_np, y_np, p, size=bootstrap_size
                )
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
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        if sample_weight is not None:
            assert len(sample_weight) == len(
                y
            ), "Data length must be equal for sample_weight and the input data"
            sample_weight = to_numpy(sample_weight)

        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=to_numpy(X), treatment=treatment_np, y=y_np)
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

    def predict(self, X, p=None):
        X_np = to_numpy(X)
        te = np.zeros((X_np.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = self.models_tau[group].predict(X)
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
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        sample_weight = (
            to_numpy(sample_weight) if sample_weight is not None else np.ones(len(y_np))
        )
        assert len(sample_weight) == len(
            y_np
        ), "Data length must be equal for sample_weight and the input data"

        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=to_numpy(X), treatment=treatment_np, y=y_np)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info("generating out-of-fold CV outcome estimates")
        yhat = cross_val_predict(self.model_mu, X, y_np, cv=self.cv, n_jobs=-1)

        for group in self.t_groups:
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt_np = treatment_np[mask]
            w = (treatment_filt_np == group).astype(int)

            X_filt = filter_mask(X, mask)
            y_filt = y_np[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            sample_weight_filt = sample_weight[mask]

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
            self.vars_c[group] = get_weighted_variance(
                diff_c, sample_weight_filt[w == 0]
            )
            self.vars_t[group] = get_weighted_variance(
                diff_t, sample_weight_filt[w == 1]
            )
