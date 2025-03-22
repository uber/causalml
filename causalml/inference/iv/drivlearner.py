import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from causalml.inference.meta.explainer import Explainer
from causalml.inference.meta.utils import (
    check_treatment_vector,
    check_p_conditions,
    convert_pd_to_np,
)
from causalml.metrics import regression_metrics
from causalml.propensity import compute_propensity_score
from scipy.stats import norm
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

logger = logging.getLogger("causalml")


class BaseDRIVLearner:
    """A parent class for DRIV-learner regressor classes.

    A DRIV-learner estimates endogenous treatment effects for compliers with machine learning models.

    Details of DR-learner are available at `Kennedy (2020) <https://arxiv.org/abs/2004.14497>`_.
    The DR moment condition for LATE comes from
    `Chernozhukov et al (2018) <https://academic.oup.com/ectj/article/21/1/C1/5056401>`_.
    """

    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        """Initialize a DR-learner.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects in both the control and treatment
                groups
            control_outcome_learner (optional): a model to estimate outcomes in the control group
            treatment_outcome_learner (optional): a model to estimate outcomes in the treatment group
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group. It needs
                to take `sample_weight` as an input argument in `fit()`.
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        assert (learner is not None) or (
            (control_outcome_learner is not None)
            and (treatment_outcome_learner is not None)
            and (treatment_effect_learner is not None)
        )

        if control_outcome_learner is None:
            self.model_mu_c = deepcopy(learner)
        else:
            self.model_mu_c = control_outcome_learner

        if treatment_outcome_learner is None:
            self.model_mu_t = deepcopy(learner)
        else:
            self.model_mu_t = treatment_outcome_learner

        if treatment_effect_learner is None:
            self.model_tau = deepcopy(learner)
        else:
            self.model_tau = treatment_effect_learner

        self.ate_alpha = ate_alpha
        self.control_name = control_name

        self.propensity_1 = None
        self.propensity_0 = None
        self.propensity_assign = None

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

    def fit(
        self, X, assignment, treatment, y, p=None, pZ=None, seed=None, calibrate=True
    ):
        """Fit the inference model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            assignment (np.array or pd.Series): a (0,1)-valued assignment vector. The assignment is the
                instrumental variable that does not depend on unknown confounders. The assignment status
                influences treatment in a monotonic way, i.e. one can only be more likely to take the
                treatment if assigned.
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (2-tuple of np.ndarray or pd.Series or dict, optional): The first (second) element corresponds to
                unassigned (assigned) units. Each is an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of float
                (0,1). If None will run ElasticNetPropensityModel() to generate the propensity scores.
            pZ (np.array or pd.Series, optional): an array of assignment probability of float (0,1); if None
                will run ElasticNetPropensityModel() to generate the assignment probability score.
            seed (int): random seed for cross-fitting
        """
        X, treatment, assignment, y = convert_pd_to_np(X, treatment, assignment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        # The estimator splits the data into 3 partitions for cross-fit on the propensity score estimation,
        # the outcome regression, and the treatment regression on the doubly robust estimates. The use of
        # the partitions is rotated so we do not lose on the sample size.  We do not cross-fit the assignment
        # score estimation as the assignment process is usually simple.
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        split_indices = [index for _, index in cv.split(y)]

        self.models_mu_c = {
            group: [
                deepcopy(self.model_mu_c),
                deepcopy(self.model_mu_c),
                deepcopy(self.model_mu_c),
            ]
            for group in self.t_groups
        }
        self.models_mu_t = {
            group: [
                deepcopy(self.model_mu_t),
                deepcopy(self.model_mu_t),
                deepcopy(self.model_mu_t),
            ]
            for group in self.t_groups
        }
        self.models_tau = {
            group: [
                deepcopy(self.model_tau),
                deepcopy(self.model_tau),
                deepcopy(self.model_tau),
            ]
            for group in self.t_groups
        }

        if p is None:
            self.propensity_1 = {
                group: np.zeros(y.shape[0]) for group in self.t_groups
            }  # propensity scores for those assigned
            self.propensity_0 = {
                group: np.zeros(y.shape[0]) for group in self.t_groups
            }  # propensity scores for those not assigned
        if pZ is None:
            self.propensity_assign, _ = compute_propensity_score(
                X=X,
                treatment=assignment,
                X_pred=X,
                treatment_pred=assignment,
                calibrate_p=calibrate,
            )
        else:
            self.propensity_assign = pZ

        for ifold in range(3):
            treatment_idx = split_indices[ifold]
            outcome_idx = split_indices[(ifold + 1) % 3]
            tau_idx = split_indices[(ifold + 2) % 3]

            treatment_treat, treatment_out, treatment_tau = (
                treatment[treatment_idx],
                treatment[outcome_idx],
                treatment[tau_idx],
            )
            assignment_treat, assignment_out, assignment_tau = (
                assignment[treatment_idx],
                assignment[outcome_idx],
                assignment[tau_idx],
            )
            y_out, y_tau = y[outcome_idx], y[tau_idx]
            X_treat, X_out, X_tau = X[treatment_idx], X[outcome_idx], X[tau_idx]
            pZ_tau = self.propensity_assign[tau_idx]

            if p is None:
                logger.info("Generating propensity score")
                cur_p_1 = dict()
                cur_p_0 = dict()

                for group in self.t_groups:
                    mask = (treatment_treat == group) | (
                        treatment_treat == self.control_name
                    )
                    mask_1, mask_0 = (
                        mask & (assignment_treat == 1),
                        mask & (assignment_treat == 0),
                    )
                    cur_p_1[group], _ = compute_propensity_score(
                        X=X_treat[mask_1],
                        treatment=(treatment_treat[mask_1] == group).astype(int),
                        X_pred=X_tau,
                        treatment_pred=(treatment_tau == group).astype(int),
                    )
                    if (treatment_treat[mask_0] == group).sum() == 0:
                        cur_p_0[group] = np.zeros(X_tau.shape[0])
                    else:
                        cur_p_0[group], _ = compute_propensity_score(
                            X=X_treat[mask_0],
                            treatment=(treatment_treat[mask_0] == group).astype(int),
                            X_pred=X_tau,
                            treatment_pred=(treatment_tau == group).astype(int),
                        )
                    self.propensity_1[group][tau_idx] = cur_p_1[group]
                    self.propensity_0[group][tau_idx] = cur_p_0[group]
            else:
                cur_p_1 = dict()
                cur_p_0 = dict()
                if isinstance(p[0], (np.ndarray, pd.Series)):
                    cur_p_0 = {self.t_groups[0]: convert_pd_to_np(p[0][tau_idx])}
                else:
                    cur_p_0 = {g: prop[tau_idx] for g, prop in p[0].items()}
                check_p_conditions(cur_p_0, self.t_groups)

                if isinstance(p[1], (np.ndarray, pd.Series)):
                    cur_p_1 = {self.t_groups[0]: convert_pd_to_np(p[1][tau_idx])}
                else:
                    cur_p_1 = {g: prop[tau_idx] for g, prop in p[1].items()}
                check_p_conditions(cur_p_1, self.t_groups)

            logger.info("Generate outcome regressions")
            for group in self.t_groups:
                mask = (treatment_out == group) | (treatment_out == self.control_name)
                mask_1, mask_0 = (
                    mask & (assignment_out == 1),
                    mask & (assignment_out == 0),
                )
                self.models_mu_c[group][ifold].fit(X_out[mask_0], y_out[mask_0])
                self.models_mu_t[group][ifold].fit(X_out[mask_1], y_out[mask_1])

            logger.info("Fit pseudo outcomes from the DR formula")

            for group in self.t_groups:
                mask = (treatment_tau == group) | (treatment_tau == self.control_name)
                treatment_filt = treatment_tau[mask]
                X_filt = X_tau[mask]
                y_filt = y_tau[mask]
                w_filt = (treatment_filt == group).astype(int)
                p_1_filt = cur_p_1[group][mask]
                p_0_filt = cur_p_0[group][mask]
                z_filt = assignment_tau[mask]
                pZ_filt = pZ_tau[mask]
                mu_t = self.models_mu_t[group][ifold].predict(X_filt)
                mu_c = self.models_mu_c[group][ifold].predict(X_filt)
                dr = (
                    z_filt * (y_filt - mu_t) / pZ_filt
                    - (1 - z_filt) * (y_filt - mu_c) / (1 - pZ_filt)
                    + mu_t
                    - mu_c
                )
                weight = (
                    z_filt * (w_filt - p_1_filt) / pZ_filt
                    - (1 - z_filt) * (w_filt - p_0_filt) / (1 - pZ_filt)
                    + p_1_filt
                    - p_0_filt
                )
                dr /= weight
                self.models_tau[group][ifold].fit(X_filt, dr, sample_weight=weight**2)

    def predict(self, X, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects for compliers, i.e. those individuals
                who take the treatment only if they are assigned.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        yhat_cs = {}
        yhat_ts = {}

        for i, group in enumerate(self.t_groups):
            models_tau = self.models_tau[group]
            _te = np.r_[[model.predict(X) for model in models_tau]].mean(axis=0)
            te[:, i] = np.ravel(_te)
            yhat_cs[group] = np.r_[
                [model.predict(X) for model in self.models_mu_c[group]]
            ].mean(axis=0)
            yhat_ts[group] = np.r_[
                [model.predict(X) for model in self.models_mu_t[group]]
            ].mean(axis=0)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

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
        assignment,
        treatment,
        y,
        p=None,
        pZ=None,
        return_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        return_components=False,
        verbose=True,
        seed=None,
        calibrate=True,
    ):
        """Fit the treatment effect and outcome models of the R learner and predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            assignment (np.array or pd.Series): a (0,1)-valued assignment vector. The assignment is the
                instrumental variable that does not depend on unknown confounders. The assignment status
                influences treatment in a monotonic way, i.e. one can only be more likely to take the
                treatment if assigned.
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (2-tuple of np.ndarray or pd.Series or dict, optional): The first (second) element corresponds to
                unassigned (assigned) units. Each is an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of float
                (0,1). If None will run ElasticNetPropensityModel() to generate the propensity scores.
            pZ (np.array or pd.Series, optional): an array of assignment probability of float (0,1); if None
                will run ElasticNetPropensityModel() to generate the assignment probability score.
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (str): whether to output progress logs
            seed (int): random seed for cross-fitting
        Returns:
            (numpy.ndarray): Predictions of treatment effects for compliers, , i.e. those individuals
                who take the treatment only if they are assigned. Output dim: [n_samples, n_treatment]
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        X, assignment, treatment, y = convert_pd_to_np(X, assignment, treatment, y)
        self.fit(X, assignment, treatment, y, p, seed, calibrate)

        if p is None:
            p = (self.propensity_0, self.propensity_1)
        else:
            check_p_conditions(p[0], self.t_groups)
            check_p_conditions(p[1], self.t_groups)

        if isinstance(p[0], (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = (
                {treatment_name: convert_pd_to_np(p[0])},
                {treatment_name: convert_pd_to_np(p[1])},
            )
        elif isinstance(p[0], dict):
            p = (
                {
                    treatment_name: convert_pd_to_np(_p)
                    for treatment_name, _p in p[0].items()
                },
                {
                    treatment_name: convert_pd_to_np(_p)
                    for treatment_name, _p in p[1].items()
                },
            )

        if pZ is None:
            pZ = self.propensity_assign

        te = self.predict(
            X, treatment=treatment, y=y, return_components=return_components
        )

        if not return_ci:
            return te
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(
                shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(
                    X, assignment, treatment, y, p, pZ, size=bootstrap_size, seed=seed
                )
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(
                te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2
            )

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau = deepcopy(models_tau_global)

            return (te, te_lower, te_upper)

    def estimate_ate(
        self,
        X,
        assignment,
        treatment,
        y,
        p=None,
        pZ=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        seed=None,
        calibrate=True,
    ):
        """Estimate the Average Treatment Effect (ATE) for compliers.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            assignment (np.array or pd.Series): an assignment vector. The assignment is the
                instrumental variable that does not depend on unknown confounders. The assignment status
                influences treatment in a monotonic way, i.e. one can only be more likely to take the
                treatment if assigned.
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (2-tuple of np.ndarray or pd.Series or dict, optional): The first (second) element corresponds to
                unassigned (assigned) units. Each is an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of float
                (0,1). If None will run ElasticNetPropensityModel() to generate the propensity scores.
            pZ (np.array or pd.Series, optional): an array of assignment probability of float (0,1); if None
                will run ElasticNetPropensityModel() to generate the assignment probability score.
            bootstrap_ci (bool): whether run bootstrap for confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            seed (int): random seed for cross-fitting
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        te, yhat_cs, yhat_ts = self.fit_predict(
            X,
            assignment,
            treatment,
            y,
            p,
            return_components=True,
            seed=seed,
            calibrate=calibrate,
        )
        X, assignment, treatment, y = convert_pd_to_np(X, assignment, treatment, y)

        if p is None:
            p = (self.propensity_0, self.propensity_1)
        else:
            check_p_conditions(p[0], self.t_groups)
            check_p_conditions(p[1], self.t_groups)

        if isinstance(p[0], (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = (
                {treatment_name: convert_pd_to_np(p[0])},
                {treatment_name: convert_pd_to_np(p[1])},
            )
        elif isinstance(p[0], dict):
            p = (
                {
                    treatment_name: convert_pd_to_np(_p)
                    for treatment_name, _p in p[0].items()
                },
                {
                    treatment_name: convert_pd_to_np(_p)
                    for treatment_name, _p in p[1].items()
                },
            )

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            _ate = te[:, i].mean()

            mask = (treatment == group) | (treatment == self.control_name)
            mask_1, mask_0 = mask & (assignment == 1), mask & (assignment == 0)
            Gamma = (treatment[mask_1] == group).mean() - (
                treatment[mask_0] == group
            ).mean()

            y_filt_1, y_filt_0 = y[mask_1], y[mask_0]
            yhat_0 = yhat_cs[group][mask_0]
            yhat_1 = yhat_ts[group][mask_1]
            treatment_filt_1, treatment_filt_0 = treatment[mask_1], treatment[mask_0]
            prob_treatment_1, prob_treatment_0 = (
                p[1][group][mask_1],
                p[0][group][mask_0],
            )
            w = (assignment[mask]).mean()

            part_1 = (
                (y_filt_1 - yhat_1).var()
                + _ate**2 * (treatment_filt_1 - prob_treatment_1).var()
                - 2
                * _ate
                * (y_filt_1 * treatment_filt_1 - yhat_1 * prob_treatment_1).mean()
            )
            part_0 = (
                (y_filt_0 - yhat_0).var()
                + _ate**2 * (treatment_filt_0 - prob_treatment_0).var()
                - 2
                * _ate
                * (y_filt_0 * treatment_filt_0 - yhat_0 * prob_treatment_0).mean()
            )
            part_2 = np.mean(
                (
                    yhat_ts[group][mask]
                    - yhat_cs[group][mask]
                    - _ate * (p[1][group][mask] - p[0][group][mask])
                )
                ** 2
            )

            # SE formula is based on the lower bound formula (9) from Frölich, Markus. 2006.
            # "Nonparametric IV estimation of local average treatment effects wth covariates."
            # Journal of Econometrics.
            se = np.sqrt((part_1 / w + part_0 / (1 - w)) + part_2) / Gamma

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
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_global = deepcopy(self.models_tau)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                cate_b = self.bootstrap(
                    X, assignment, treatment, y, p, pZ, size=bootstrap_size, seed=seed
                )
                ate_bootstraps[:, n] = cate_b.mean()

            ate_lower = np.percentile(
                ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1
            )
            ate_upper = np.percentile(
                ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1
            )

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau = deepcopy(models_tau_global)
            return ate, ate_lower, ate_upper

    def bootstrap(self, X, assignment, treatment, y, p, pZ, size=10000, seed=None):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]

        if isinstance(p[0], (np.ndarray, pd.Series)):
            p0_b = {self.t_groups[0]: convert_pd_to_np(p[0][idxs])}
        else:
            p0_b = {g: prop[idxs] for g, prop in p[0].items()}
        if isinstance(p[1], (np.ndarray, pd.Series)):
            p1_b = {self.t_groups[0]: convert_pd_to_np(p[1][idxs])}
        else:
            p1_b = {g: prop[idxs] for g, prop in p[1].items()}

        pZ_b = pZ[idxs]
        assignment_b = assignment[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(
            X=X_b,
            assignment=assignment_b,
            treatment=treatment_b,
            y=y_b,
            p=(p0_b, p1_b),
            pZ=pZ_b,
            seed=seed,
        )
        te_b = self.predict(X=X)
        return te_b

    def get_importance(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        method="auto",
        normalize=True,
        test_size=0.3,
        random_state=None,
    ):
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (np.array): list/array of feature names. If None, an enumerated list will be used
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
        explainer = Explainer(
            method=method,
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
        )
        return explainer.get_importance()

    def get_shap_values(self, X=None, model_tau_feature=None, tau=None, features=None):
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates shapley values.
        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
        """
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
        )
        return explainer.get_shap_values()

    def plot_importance(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        method="auto",
        normalize=True,
        test_size=0.3,
        random_state=None,
    ):
        """
        Builds a model (using X to predict estimated/actual tau), and then plots feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
        explainer = Explainer(
            method=method,
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
        )
        explainer.plot_importance()

    def plot_shap_values(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        shap_dict=None,
        **kwargs,
    ):
        """
        Plots distribution of shapley values.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix. Required if shap_dict is None.
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
        """
        override_checks = False if shap_dict is None else True
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            override_checks=override_checks,
            classes=self._classes,
        )
        explainer.plot_shap_values(shap_dict=shap_dict)

    def plot_shap_dependence(
        self,
        treatment_group,
        feature_idx,
        X,
        tau,
        model_tau_feature=None,
        features=None,
        shap_dict=None,
        interaction_idx="auto",
        **kwargs,
    ):
        """
        Plots dependency of shapley values for a specified feature, colored by an interaction feature.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        This plots the value of the feature on the x-axis and the SHAP value of the same feature
        on the y-axis. This shows how the model depends on the given feature, and is like a
        richer extension of the classical partial dependence plots. Vertical dispersion of the
        data points represents interaction effects.

        Args:
            treatment_group (str or int): name of treatment group to create dependency plot on
            feature_idx (str or int): feature index / name to create dependency plot on
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
            interaction_idx (optional, str or int): feature index / name used in coloring scheme as interaction feature.
                If "auto" then shap.common.approximate_interactions is used to pick what seems to be the
                strongest interaction (note that to find to true strongest interaction you need to compute
                the SHAP interaction values).
        """
        override_checks = False if shap_dict is None else True
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            override_checks=override_checks,
            classes=self._classes,
        )
        explainer.plot_shap_dependence(
            treatment_group=treatment_group,
            feature_idx=feature_idx,
            shap_dict=shap_dict,
            interaction_idx=interaction_idx,
            **kwargs,
        )


class BaseDRIVRegressor(BaseDRIVLearner):
    """
    A parent class for DRIV-learner regressor classes.
    """

    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        """Initialize a DRIV-learner regressor.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects in both the control and treatment
                groups
            control_outcome_learner (optional): a model to estimate outcomes in the control group
            treatment_outcome_learner (optional): a model to estimate outcomes in the treatment group
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group. It needs
                to take `sample_weight` as an input argument in `fit()`.
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
        )


class XGBDRIVRegressor(BaseDRIVRegressor):
    def __init__(self, ate_alpha=0.05, control_name=0, *args, **kwargs):
        """Initialize a DRIV-learner with two XGBoost models."""
        super().__init__(
            learner=XGBRegressor(*args, **kwargs),
            ate_alpha=ate_alpha,
            control_name=control_name,
        )
