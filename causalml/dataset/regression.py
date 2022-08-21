import logging

import numpy as np
from scipy.special import expit, logit

logger = logging.getLogger("causalml")


def synthetic_data(mode=1, n=1000, p=5, sigma=1.0, adj=0.0):
    """ Synthetic data in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        mode (int, optional): mode of the simulation: \
            1 for difficult nuisance components and an easy treatment effect. \
            2 for a randomized trial. \
            3 for an easy propensity and a difficult baseline. \
            4 for unrelated treatment and control groups. \
            5 for a hidden confounder biasing treatment.
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
                     It does not apply to mode == 2 or 3.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    catalog = {
        1: simulate_nuisance_and_easy_treatment,
        2: simulate_randomized_trial,
        3: simulate_easy_propensity_difficult_baseline,
        4: simulate_unrelated_treatment_control,
        5: simulate_hidden_confounder,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    return catalog[mode](n, p, sigma, adj)


def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_randomized_trial(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data of a randomized trial
        From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_easy_propensity_difficult_baseline(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with easy propensity and a difficult baseline
        From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
    e = 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
    tau = np.repeat(1.0, n)

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_unrelated_treatment_control(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with unrelated treatment and control groups.
        From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = (
        np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2])
        + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
    ) / 2
    e = 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))
    e = expit(logit(e) - adj)
    tau = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) - np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_hidden_confounder(n=10000, p=5, sigma=1.0, adj=0.0):
    """Synthetic dataset with a hidden confounder biasing treatment.
        From Louizos et al. (2018) "Causal Effect Inference with Deep Latent-Variable Models"
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    z = np.random.binomial(1, 0.5, size=n).astype(np.double)
    X = np.random.normal(z, 5 * z + 3 * (1 - z), size=(p, n)).T
    e = 0.75 * z + 0.25 * (1 - z)
    w = np.random.binomial(1, e)
    b = expit(3 * (z + 2 * (2 * w - 2)))
    y = np.random.binomial(1, b)

    # Compute true ite tau for evaluation (via Monte Carlo approximation).
    t0_t1 = np.array([[0.0], [1.0]])
    y_t0, y_t1 = expit(3 * (z + 2 * (2 * t0_t1 - 2)))
    tau = y_t1 - y_t0
    return y, X, w, tau, b, e
