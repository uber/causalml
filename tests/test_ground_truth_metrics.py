"""Tests for the metrics that score against a known ground truth.

These need no dataset: `synthetic_data` returns the individual effect `tau`, and
the properties asserted here are exact ones the definitions imply.
"""

import numpy as np
import pytest

from causalml.dataset import synthetic_data
from causalml.metrics import ate_error, pehe, policy_risk

from .const import RANDOM_SEED


@pytest.fixture
def tau():
    np.random.seed(RANDOM_SEED)
    _, _, _, tau, _, _ = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)
    return tau


def test_pehe_of_the_truth_is_zero(tau):
    """A perfect estimate has no error, squared or not."""
    assert pehe(tau, tau) == 0.0
    assert pehe(tau, tau, squared=False) == 0.0


def test_pehe_of_a_constant_predictor_is_the_variance_of_tau(tau):
    """Predicting the mean effect everywhere leaves exactly the heterogeneity.

    This is the property that makes PEHE worth reporting next to an ATE error: a
    model can get the average exactly right and still score the full variance of
    the individual effects.
    """
    constant = np.full_like(tau, tau.mean())

    assert pehe(tau, constant) == pytest.approx(tau.var())
    assert pehe(tau, constant, squared=False) == pytest.approx(tau.std())
    assert ate_error(tau, constant) == pytest.approx(0.0, abs=1e-12)


def test_pehe_root_is_the_square_root_of_the_squared_form(tau):
    """`squared=False` is the root, on the same scale as the effect."""
    noisy = tau + np.random.RandomState(RANDOM_SEED).normal(size=tau.shape)

    assert pehe(tau, noisy, squared=False) == pytest.approx(np.sqrt(pehe(tau, noisy)))


def test_pehe_rejects_mismatched_lengths(tau):
    """Comparing different numbers of units is a caller error, not a mean."""
    with pytest.raises(ValueError, match="same length"):
        pehe(tau, tau[:-1])


def test_ate_error_is_the_shift_of_a_shifted_estimate(tau):
    """Shifting every prediction by a constant moves the ATE error by exactly it."""
    assert ate_error(tau, tau + 0.75) == pytest.approx(0.75)
    assert ate_error(tau, tau - 0.75) == pytest.approx(0.75)


def test_ate_error_accepts_scalars(tau):
    """An experiment gives one number, not a per-unit truth."""
    assert ate_error(2.0, 2.5) == pytest.approx(0.5)


def test_policy_risk_prefers_the_oracle_policy_to_a_random_one():
    """Targeting by the true effect beats targeting at random.

    The outcome is constructed so treating a unit with a positive effect helps and
    treating one with a negative effect hurts, which is the ordering the metric
    exists to detect.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n = 4000
    tau = rng.choice([-0.4, 0.4], size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    baseline = 0.5
    y = np.clip(baseline + treatment * tau + rng.normal(scale=0.01, size=n), 0, 1)

    oracle = policy_risk(y, treatment, tau)
    random_policy = policy_risk(y, treatment, rng.normal(size=n))

    assert oracle < random_policy
    # The oracle treats the half it helps and leaves the rest alone, so its value
    # is 0.5 * (baseline + 0.4) + 0.5 * baseline rather than baseline + 0.4: the
    # untreated half still contributes its own outcome to the policy's value.
    expected_value = 0.5 * (baseline + 0.4) + 0.5 * baseline
    assert oracle == pytest.approx(1 - expected_value, abs=0.02)


def test_policy_risk_raises_when_an_arm_is_empty():
    """With nobody assigned as the policy recommends, there is nothing to average."""
    y = np.array([1.0, 0.0, 1.0, 0.0])
    treatment = np.zeros(4, dtype=int)
    tau_hat = np.ones(4)

    with pytest.raises(ValueError, match="both arms"):
        policy_risk(y, treatment, tau_hat)
