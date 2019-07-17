from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np


logger = logging.getLogger('causalml')


def synthetic_data(mode=1, n=1000, p=5, sigma=1.0):
    ''' Synthetic data in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

    Args:
        mode (int, optional): mode of the simulation: \
            1 for a difficult nuisance components and an easy treatment effect. \
            2 for a randomized trial. \
            3 for an easy propensity and a difficult baseline. \
            4 for unrelated treatment and control groups.
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:

            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    '''
    assert mode in (1, 2, 3, 4), 'Invalid mode {}. Should be between 1 and 4'.format(mode)

    catalog = {1: simulate_nuisance_and_easy_treatment,
               2: simulate_randomized_trial,
               3: simulate_easy_propensity_difficult_baseline,
               4: simulate_unrelated_treatment_control}

    return catalog[mode](n, p, sigma)


def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0):
    ''' Synthetic data with a diffult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:

            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    '''

    X = np.random.uniform(size=n*p).reshape((n,-1))
    b = np.sin(np.pi * X[:,0] * X[:,1]) + 2 * (X[:,2] - 0.5) ** 2 + X[:,3] + 0.5 * X[:,4]
    eta = 0.1
    e = np.maximum(np.repeat(eta, n), np.minimum(np.sin(np.pi * X[:,0] * X[:,1]), np.repeat(1-eta, n)))
    tau = (X[:,0] + X[:,1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_randomized_trial(n=1000, p=5, sigma=1.0):
    ''' Synthetic data of a randomized trial
        From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:

            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    '''

    X = np.random.normal(size=n*p).reshape((n,-1))
    b = np.maximum(np.repeat(0.0, n), X[:,0] + X[:,1] + X[:,2]) + np.maximum(np.repeat(0.0, n), X[:,3] + X[:,4])
    e = np.repeat(0.5, n)
    tau = X[:,0] + np.log1p(np.exp(X[:,1]))

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_easy_propensity_difficult_baseline(n=1000, p=5, sigma=1.0):
    ''' Synthetic data with easy propensity and a difficult baseline
        From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:

            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    '''

    X = np.random.normal(size=n*p).reshape((n,-1))
    b = 2 * np.log1p(np.exp(X[:,0] + X[:,1] + X[:,2]))
    e = 1/(1 + np.exp(X[:,1] + X[:,2]))
    tau = np.repeat(1.0, n)

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_unrelated_treatment_control(n=1000, p=5, sigma=1.0):
    ''' Synthetic data with unrelated treatment and control groups.
        From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'

    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:

            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    '''

    X = np.random.normal(size=n*p).reshape((n,-1))
    b = (np.maximum(np.repeat(0.0, n), X[:,0] + X[:,1] + X[:,2]) + np.maximum(np.repeat(0.0, n), X[:,3] + X[:,4])) / 2
    e = 1/(1 + np.exp(-X[:,0]) + np.exp(-X[:,1]))
    tau = np.maximum(np.repeat(0.0, n), X[:,0] + X[:,1] + X[:,2]) - np.maximum(np.repeat(0.0, n), X[:,3] + X[:,4])

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e

