from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from causalml.inference.meta.base import BaseLearner
from causalml.inference.meta.utils import (
    check_treatment_vector,
    check_p_conditions,
    filter_mask,
    filter_index,
    to_numpy,
)
from causalml.metrics import regression_metrics, classification_metrics
from causalml.propensity import compute_propensity_score

logger = logging.getLogger("causalml")


class BaseDRLearner(BaseLearner):
    """A parent class for DR-learner regressor classes."""

    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        assert (learner is not None) or (
            (control_outcome_learner is not None)
            and (treatment_outcome_learner is not None)
            and (treatment_effect_learner is not None)
        )

        self.model_mu_c = (
            deepcopy(learner)
            if control_outcome_learner is None
            else control_outcome_learner
        )
        self.model_mu_t = (
            deepcopy(learner)
            if treatment_outcome_learner is None
            else treatment_outcome_learner
        )
        self.model_tau = (
            deepcopy(learner)
            if treatment_effect_learner is None
            else treatment_effect_learner
        )

        self.ate_alpha = ate_alpha
        self.control_name = control_name
        self.propensity = None

    def __repr__(self):
        return (
            "{}(control_outcome_learner={},\n"
            "\ttreatment_outcome_learner={},\n"
            "\ttreatment_effect_learner={})".format(
                self.__class__.__name__,
                self.model_mu_c.__repr__(),
                self.model_mu_t.__repr__(),
                self.model_tau.__repr__(),
            )
        )

    def fit(self, X, treatment, y, p=None, seed=None):
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        split_indices = [index for _, index in cv.split(y_np)]

        self.models_mu_c = [deepcopy(self.model_mu_c) for _ in range(3)]
        self.models_mu_t = {
            group: [deepcopy(self.model_mu_t) for _ in range(3)]
            for group in self.t_groups
        }
        self.models_tau = {
            group: [deepcopy(self.model_tau) for _ in range(3)]
            for group in self.t_groups
        }

        if p is None:
            self.propensity = {
                group: np.zeros(y_np.shape[0]) for group in self.t_groups
            }

        for ifold in range(3):
            treatment_idx = split_indices[ifold]
            outcome_idx = split_indices[(ifold + 1) % 3]
            tau_idx = split_indices[(ifold + 2) % 3]

            treatment_treat = filter_index(treatment, treatment_idx)
            treatment_out = filter_index(treatment, outcome_idx)
            treatment_tau = filter_index(treatment, tau_idx)

            treatment_treat_np = to_numpy(treatment_treat)
            treatment_out_np = to_numpy(treatment_out)
            treatment_tau_np = to_numpy(treatment_tau)

            y_out = y_np[outcome_idx]
            y_tau = y_np[tau_idx]

            X_treat = filter_index(X, treatment_idx)
            X_out = filter_index(X, outcome_idx)
            X_tau = filter_index(X, tau_idx)

            if p is None:
                logger.info("Generating propensity score")
                cur_p = dict()
                for group in self.t_groups:
                    mask = (treatment_treat_np == group) | (
                        treatment_treat_np == self.control_name
                    )
                    treatment_filt = filter_mask(treatment_treat, mask)
                    X_filt = filter_mask(X_treat, mask)
                    w_filt = (to_numpy(treatment_filt) == group).astype(int)
                    w = (treatment_tau_np == group).astype(int)
                    cur_p[group], _ = compute_propensity_score(
                        X=X_filt, treatment=w_filt, X_pred=X_tau, treatment_pred=w
                    )
                    self.propensity[group][tau_idx] = cur_p[group]
            else:
                cur_p = dict()
                if isinstance(p, (np.ndarray, pd.Series)):
                    cur_p = {self.t_groups[0]: to_numpy(filter_index(p, tau_idx))}
                else:
                    cur_p = {
                        g: to_numpy(filter_index(prop, tau_idx))
                        for g, prop in p.items()
                    }
                check_p_conditions(cur_p, self.t_groups)

            logger.info("Generate outcome regressions")
            self.models_mu_c[ifold].fit(
                filter_mask(X_out, treatment_out_np == self.control_name),
                y_out[treatment_out_np == self.control_name],
            )
            for group in self.t_groups:
                self.models_mu_t[group][ifold].fit(
                    filter_mask(X_out, treatment_out_np == group),
                    y_out[treatment_out_np == group],
                )

            logger.info("Fit pseudo outcomes from the DR formula")
            for group in self.t_groups:
                mask = (treatment_tau_np == group) | (
                    treatment_tau_np == self.control_name
                )
                treatment_filt_np = treatment_tau_np[mask]
                X_filt = filter_mask(X_tau, mask)
                y_filt = y_tau[mask]
                w_filt = (treatment_filt_np == group).astype(int)
                p_filt = cur_p[group][mask]
                mu_t = self.models_mu_t[group][ifold].predict(X_filt)
                mu_c = self.models_mu_c[ifold].predict(X_filt)
                dr = (
                    (w_filt - p_filt)
                    / p_filt
                    / (1 - p_filt)
                    * (y_filt - mu_t * w_filt - mu_c * (1 - w_filt))
                    + mu_t
                    - mu_c
                )
                self.models_tau[group][ifold].fit(X_filt, dr)

    def bootstrap(self, X, treatment, y, p=None, size=10000, rng=None, seed=None):
        if rng is not None:
            idxs = rng.choice(np.arange(0, to_numpy(X).shape[0]), size=size)
        else:
            idxs = np.random.choice(np.arange(0, to_numpy(X).shape[0]), size=size)
        X_b = filter_index(X, idxs)
        p_b = {group: _p[idxs] for group, _p in p.items()} if p is not None else None
        treatment_b = filter_index(treatment, idxs)
        y_b = to_numpy(y)[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b, seed=seed)
        return self.predict(X=X, p=p)

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        X_np = to_numpy(X)
        te = np.zeros((X_np.shape[0], self.t_groups.shape[0]))
        yhat_cs = {}
        yhat_ts = {}

        for i, group in enumerate(self.t_groups):
            _te = np.r_[[model.predict(X) for model in self.models_tau[group]]].mean(
                axis=0
            )
            te[:, i] = np.ravel(_te)
            yhat_cs[group] = np.r_[
                [model.predict(X) for model in self.models_mu_c]
            ].mean(axis=0)
            yhat_ts[group] = np.r_[
                [model.predict(X) for model in self.models_mu_t[group]]
            ].mean(axis=0)

            if (y is not None) and (treatment is not None) and verbose:
                treatment_np = to_numpy(treatment)
                mask = (treatment_np == group) | (treatment_np == self.control_name)
                treatment_filt_np = treatment_np[mask]
                y_filt = to_numpy(filter_mask(y, mask))
                w = (treatment_filt_np == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = yhat_cs[group][mask][w == 0]
                yhat[w == 1] = yhat_ts[group][mask][w == 1]

                logger.info("Error metrics for group {}".format(group))
                regression_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, yhat_cs, yhat_ts

    def fit_predict(
        self,
        X,
        treatment,
        y,
        p=None,
        return_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        return_components=False,
        verbose=True,
        seed=None,
    ):
        self.fit(X, treatment, y, p, seed)

        if p is None:
            p = self.propensity

        check_p_conditions(p, self.t_groups)
        if isinstance(p, (np.ndarray, pd.Series)):
            p = {self.t_groups[0]: to_numpy(p)}
        elif isinstance(p, dict):
            p = {k: to_numpy(v) for k, v in p.items()}

        te = self.predict(
            X, treatment=treatment, y=y, return_components=return_components
        )

        if not return_ci:
            return te
        else:
            X_np = to_numpy(X)
            treatment_np = to_numpy(treatment)
            y_np = to_numpy(y)

            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(
                shape=(X_np.shape[0], self.t_groups.shape[0], n_bootstraps)
            )
            rng = np.random.default_rng(seed) if seed is not None else None

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                bootstrap_seed = (
                    int(rng.integers(np.iinfo(np.int32).max))
                    if rng is not None
                    else None
                )
                te_b = self.bootstrap(
                    X_np,
                    treatment_np,
                    y_np,
                    p,
                    size=bootstrap_size,
                    rng=rng,
                    seed=bootstrap_seed,
                )
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(
                te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2
            )

            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau = deepcopy(models_tau_global)

            return (te, te_lower, te_upper)

    def estimate_ate(
        self,
        X,
        treatment,
        y,
        p=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        seed=None,
        pretrain=False,
    ):
        if pretrain:
            te, yhat_cs, yhat_ts = self.predict(
                X, treatment, y, p, return_components=True
            )
        else:
            te, yhat_cs, yhat_ts = self.fit_predict(
                X, treatment, y, p, return_components=True, seed=seed
            )

        treatment_np = to_numpy(treatment)
        y_np = to_numpy(y)

        if p is None:
            p = self.propensity
        else:
            check_p_conditions(p, self.t_groups)
        if isinstance(p, (np.ndarray, pd.Series)):
            p = {self.t_groups[0]: to_numpy(p)}
        elif isinstance(p, dict):
            p = {k: to_numpy(v) for k, v in p.items()}

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            _ate = te[:, i].mean()
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = treatment_np[mask]
            w = (treatment_filt == group).astype(int)
            prob_treatment = float(sum(w)) / w.shape[0]

            yhat_c = yhat_cs[group][mask]
            yhat_t = yhat_ts[group][mask]
            y_filt = y_np[mask]

            se = np.sqrt(
                (
                    (y_filt[w == 0] - yhat_c[w == 0]).var() / (1 - prob_treatment)
                    + (y_filt[w == 1] - yhat_t[w == 1]).var() / prob_treatment
                    + (yhat_t - yhat_c).var()
                )
                / y_filt.shape[0]
            )

            _ate_lb = _ate - se * norm.ppf(1 - self.ate_alpha / 2)
            _ate_ub = _ate + se * norm.ppf(1 - self.ate_alpha / 2)

            ate[i] = _ate
            ate_lb[i] = _ate_lb
            ate_ub[i] = _ate_ub

        if not bootstrap_ci:
            return ate, ate_lb, ate_ub
        else:
            X_np = to_numpy(X)
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_global = deepcopy(self.models_tau)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))
            rng = np.random.default_rng(seed) if seed is not None else None

            for n in tqdm(range(n_bootstraps)):
                bootstrap_seed = (
                    int(rng.integers(np.iinfo(np.int32).max))
                    if rng is not None
                    else None
                )
                cate_b = self.bootstrap(
                    X_np,
                    treatment_np,
                    y_np,
                    p,
                    size=bootstrap_size,
                    rng=rng,
                    seed=bootstrap_seed,
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
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau = deepcopy(models_tau_global)
            return ate, ate_lower, ate_upper


class BaseDRRegressor(BaseDRLearner):
    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        super().__init__(
            learner=learner,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
        )


class BaseDRClassifier(BaseDRLearner):
    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        super().__init__(
            learner=learner,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
        )

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        X_np = to_numpy(X)
        te = np.zeros((X_np.shape[0], self.t_groups.shape[0]))
        yhat_cs = {}
        yhat_ts = {}

        for i, group in enumerate(self.t_groups):
            _te = np.r_[[model.predict(X) for model in self.models_tau[group]]].mean(
                axis=0
            )
            te[:, i] = np.ravel(_te)
            yhat_cs[group] = np.r_[
                [model.predict_proba(X)[:, 1] for model in self.models_mu_c]
            ].mean(axis=0)
            yhat_ts[group] = np.r_[
                [model.predict_proba(X)[:, 1] for model in self.models_mu_t[group]]
            ].mean(axis=0)

            if (y is not None) and (treatment is not None) and verbose:
                treatment_np = to_numpy(treatment)
                mask = (treatment_np == group) | (treatment_np == self.control_name)
                treatment_filt_np = treatment_np[mask]
                y_filt = to_numpy(filter_mask(y, mask))
                w = (treatment_filt_np == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = yhat_cs[group][mask][w == 0]
                yhat[w == 1] = yhat_ts[group][mask][w == 1]

                logger.info("Error metrics for group {}".format(group))
                classification_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, yhat_cs, yhat_ts


class XGBDRRegressor(BaseDRRegressor):
    def __init__(self, ate_alpha=0.05, control_name=0, *args, **kwargs):
        super().__init__(
            learner=XGBRegressor(*args, **kwargs),
            ate_alpha=ate_alpha,
            control_name=control_name,
        )
