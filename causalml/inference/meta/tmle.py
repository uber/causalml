import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from causalml.inference.meta.utils import (
    check_treatment_vector,
    check_p_conditions,
    convert_pd_to_np,
)
from causalml.propensity import calibrate


logger = logging.getLogger("causalml")


def logit_tmle(x, y, a, h0, h1):
    p = expit(a + x[0] * h0 + x[1] * h1)
    return np.mean(-np.log(np.power(p, y) * np.power(1 - p, 1 - y)))


def logit_tmle_grad(x, y, a, h0, h1):
    p = expit(a + x[0] * h0 + x[1] * h1)
    return np.array([-np.mean((y - p) * h0), -np.mean((y - p) * h1)])


def logit_tmle_hess(x, y, a, h0, h1):
    p = expit(a + x[0] * h0 + x[1] * h1)
    return np.array(
        [
            [np.mean(p * (1 - p) * h0 * h0), np.mean(p * (1 - p) * h0 * h1)],
            [np.mean(p * (1 - p) * h0 * h1), np.mean(p * (1 - p) * h1 * h1)],
        ]
    )


def simple_tmle(y, w, q0w, q1w, p, alpha=0.0001):
    """Calculate the ATE and variances with the simplified TMLE method.

    Args:
        y (numpy.array): an outcome vector
        w (numpy.array): a treatment vector
        q0w (numpy.array): an outcome prediction vector given no treatment
        q1w (numpy.array): an outcome prediction vector given treatment
        p (numpy.array): a propensity score vector
        alpha (float, optional): a clipping threshold for predictions

    Returns:
        (tuple)

            - ate (float): ATE
            - se (float): The standard error of ATE
    """
    scaler = MinMaxScaler()
    ystar = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    q0 = np.clip(scaler.transform(q0w.reshape(-1, 1)).flatten(), alpha, 1 - alpha)
    q1 = np.clip(scaler.transform(q1w.reshape(-1, 1)).flatten(), alpha, 1 - alpha)
    qaw = q0 * (1 - w) + q1 * w
    intercept = logit(qaw)

    h1 = w / p
    h0 = (1 - w) / (1 - p)
    sol = minimize(
        logit_tmle,
        np.zeros(2),
        args=(ystar, intercept, h0, h1),
        method="Newton-CG",
        jac=logit_tmle_grad,
        hess=logit_tmle_hess,
    )

    qawstar = scaler.inverse_transform(
        expit(intercept + sol.x[0] * h0 + sol.x[1] * h1).reshape(-1, 1)
    ).flatten()
    q0star = scaler.inverse_transform(
        expit(logit(q0) + sol.x[0] / (1 - p)).reshape(-1, 1)
    ).flatten()
    q1star = scaler.inverse_transform(
        expit(logit(q1) + sol.x[1] / p).reshape(-1, 1)
    ).flatten()

    ic = (
        (w / p - (1 - w) / (1 - p)) * (y - qawstar)
        + q1star
        - q0star
        - np.mean(q1star - q0star)
    )

    return np.mean(q1star - q0star), np.sqrt(np.var(ic) / np.size(y))


class TMLELearner(object):
    """Targeted maximum likelihood estimation.

    Ref: Gruber, S., & Van Der Laan, M. J. (2009). Targeted maximum likelihood estimation: A gentle introduction.
    """

    def __init__(
        self,
        learner,
        ate_alpha=0.05,
        control_name=0,
        cv=None,
        calibrate_propensity=True,
    ):
        """Initialize a TMLE learner.

        Args:
            learner: a model to estimate the outcome
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): the name of the control group
            cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        """
        self.model_tau = learner
        self.ate_alpha = ate_alpha
        self.control_name = control_name
        self.cv = cv
        self.calibrate_propensity = calibrate_propensity

    def __repr__(self):
        return "{}(model={}, cv={})".format(
            self.__class__.__name__, self.model_tau.__repr__(), self.cv
        )

    def estimate_ate(self, X, treatment, y, p, segment=None, return_ci=False):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict): an array of propensity scores of float (0,1) in the single-treatment
                case; or, a dictionary of treatment groups that map to propensity vectors of float (0,1)
            segment (np.array, optional): An optional segment vector of int. If given, the ATE and its CI will be
                                          estimated for each segment.
            return_ci (bool, optional): Whether to return confidence intervals

        Returns:
            (tuple): The ATE and its confidence interval (LB, UB) for each treatment, t and segment, s
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        check_p_conditions(p, self.t_groups)
        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {
                treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()
            }

        ate = []
        ate_lb = []
        ate_ub = []

        for i, group in enumerate(self.t_groups):
            logger.info("Estimating ATE for group {}.".format(group))
            w_group = (treatment == group).astype(int)
            p_group = p[group]

            if self.calibrate_propensity:
                logger.info("Calibrating propensity scores.")
                p_group = calibrate(p_group, w_group)

            yhat_c = np.zeros_like(y, dtype=float)
            yhat_t = np.zeros_like(y, dtype=float)
            if self.cv:
                for i_fold, (i_trn, i_val) in enumerate(self.cv.split(X, y), 1):
                    logger.info("Training an outcome model for CV #{}".format(i_fold))
                    self.model_tau.fit(
                        np.hstack((X[i_trn], w_group[i_trn].reshape(-1, 1))), y[i_trn]
                    )

                    yhat_c[i_val] = self.model_tau.predict(
                        np.hstack((X[i_val], np.zeros((len(i_val), 1))))
                    )
                    yhat_t[i_val] = self.model_tau.predict(
                        np.hstack((X[i_val], np.ones((len(i_val), 1))))
                    )

            else:
                self.model_tau.fit(np.hstack((X, w_group.reshape(-1, 1))), y)

                yhat_c = self.model_tau.predict(np.hstack((X, np.zeros((len(y), 1)))))
                yhat_t = self.model_tau.predict(np.hstack((X, np.ones((len(y), 1)))))

            if segment is None:
                logger.info("Training the TMLE learner.")
                _ate, se = simple_tmle(y, w_group, yhat_c, yhat_t, p_group)
                _ate_lb = _ate - se * norm.ppf(1 - self.ate_alpha / 2)
                _ate_ub = _ate + se * norm.ppf(1 - self.ate_alpha / 2)
            else:
                assert (
                    segment.shape[0] == X.shape[0] and segment.ndim == 1
                ), "Segment must be the 1-d np.array of int."
                segments = np.unique(segment)

                _ate = []
                _ate_lb = []
                _ate_ub = []
                for s in sorted(segments):
                    logger.info("Training the TMLE learner for segment {}.".format(s))
                    filt = (segment == s) & (yhat_c < np.quantile(yhat_c, q=0.99))
                    _ate_s, se = simple_tmle(
                        y[filt],
                        w_group[filt],
                        yhat_c[filt],
                        yhat_t[filt],
                        p_group[filt],
                    )
                    _ate_lb_s = _ate_s - se * norm.ppf(1 - self.ate_alpha / 2)
                    _ate_ub_s = _ate_s + se * norm.ppf(1 - self.ate_alpha / 2)

                    _ate.append(_ate_s)
                    _ate_lb.append(_ate_lb_s)
                    _ate_ub.append(_ate_ub_s)

            ate.append(_ate)
            ate_lb.append(_ate_lb)
            ate_ub.append(_ate_ub)

        return np.array(ate), np.array(ate_lb), np.array(ate_ub)
