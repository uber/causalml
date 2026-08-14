"""Metrics that score a CATE estimate against a known ground truth.

The rest of :mod:`causalml.metrics` scores how well a model *ranks* units — AUUC,
Qini, RATE — because on real data the individual effect is never observed. Where
it is observed, by simulation or by a design like Twins where both twins are seen,
these compare the estimate against it directly.

Each one names what it needs, and none of them can be computed on ordinary
observational data:

- :func:`pehe` and :func:`ate_error` need the per-unit effect ``tau``.
- :func:`policy_risk` needs randomized treatment.
"""

import numpy as np

__all__ = ["pehe", "ate_error", "policy_risk"]


def pehe(tau, tau_hat, squared=True):
    """Precision in estimating heterogeneous effects (Hill, 2011).

    The mean squared error of the individual treatment effect::

        PEHE = mean((tau_hat - tau) ** 2)

    Papers report both this and its square root; the root is on the same scale as
    the effect itself, which makes it the easier one to read.

    Only defined where ``tau`` is known per unit, which means simulated or
    semi-synthetic data. Computing it against another model's predictions
    measures agreement between two models, not accuracy.

    Args:
        tau (np.ndarray): the true individual treatment effect
        tau_hat (np.ndarray): the estimated individual treatment effect
        squared (bool): if False, return the root

    Returns:
        float, the (root) mean squared error of the individual effect
    """
    tau, tau_hat = np.asarray(tau).ravel(), np.asarray(tau_hat).ravel()
    if tau.shape != tau_hat.shape:
        raise ValueError(
            f"tau and tau_hat must have the same length, got {tau.shape} and "
            f"{tau_hat.shape}"
        )
    error = np.mean((tau_hat - tau) ** 2)
    return float(error if squared else np.sqrt(error))


def ate_error(tau, tau_hat):
    """Absolute error in the average treatment effect, usually written eps_ATE.

    ``abs(mean(tau_hat) - mean(tau))``. A model can order units well and still be
    biased on the average, and vice versa, so this is not implied by PEHE or by
    AUUC.

    On an experiment with no per-unit ground truth, pass the experimental
    estimate as ``tau``: the comparison is then against the number the experiment
    licenses rather than against a per-unit truth that does not exist.

    Args:
        tau (np.ndarray or float): the true individual effects, or the true ATE
        tau_hat (np.ndarray or float): the estimated individual effects, or the
            estimated ATE

    Returns:
        float, the absolute difference of the two averages
    """
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def policy_risk(y, treatment, tau_hat, control_name=0):
    """Expected loss of the treat-if-``tau_hat``-is-positive policy.

    Following Shalit, Johansson and Sontag (2017)::

        R_pol = 1 - (E[Y(1) | pi = 1] P(pi = 1) + E[Y(0) | pi = 0] P(pi = 0))

    where ``pi`` treats a unit when its estimated effect is positive. Each
    conditional expectation is estimated from the units that were actually
    assigned that way, which is why the treatment has to be **randomized**: on
    observational data those subgroups differ for reasons the policy did not
    choose, and the number stops describing the policy.

    The outcome is assumed to be a benefit in [0, 1], as employment is on Jobs.
    For a cost, or a scale other than [0, 1], the complement to 1 is not
    meaningful and the policy value itself is the quantity to report.

    Args:
        y (np.ndarray): the observed outcome
        treatment (np.ndarray): the treatment assignment
        tau_hat (np.ndarray): the estimated individual treatment effect
        control_name: the value of ``treatment`` marking a control unit

    Returns:
        float, one minus the value of the policy

    Raises:
        ValueError: if either arm of the policy has no units to estimate from
    """
    y, treatment, tau_hat = (
        np.asarray(y).ravel(),
        np.asarray(treatment).ravel(),
        np.asarray(tau_hat).ravel(),
    )
    treat = tau_hat > 0
    is_control = treatment == control_name

    treated_and_recommended = treat & ~is_control
    control_and_not_recommended = ~treat & is_control
    if not treated_and_recommended.any() or not control_and_not_recommended.any():
        raise ValueError(
            "policy_risk needs units assigned the way the policy recommends in "
            "both arms; one of them is empty, so its outcome cannot be estimated"
        )

    p_treat = treat.mean()
    value = y[treated_and_recommended].mean() * p_treat + y[
        control_and_not_recommended
    ].mean() * (1 - p_treat)
    return float(1 - value)
