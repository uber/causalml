import numpy as np
import pandas as pd


def get_pns_bounds(data_exp, data_obs, T, Y, type="PNS"):
    """
    Args
    ----
    data_exp : DataFrame
        Data from an experiment.
    data_obs : DataFrame
        Data from an observational study
    T : str
        Name of the binary treatment indicator
    y : str
        Name of the binary outcome indicator
    'type' : str
        Type of probability of causation desired. Acceptable args are:
        * 'PNS': Probability of necessary and sufficient causation
        * 'PS': Probability of sufficient causation
        * 'PN': Probability of necessary causation

    Notes
    -----
    Based on Equation (24) in Tian and Pearl: https://ftp.cs.ucla.edu/pub/stat_ser/r271-A.pdf

    To capture the counterfactual notation, we use `1' and `0' to indicate the actual and
    counterfactual values of a variable, respectively, and we use `do' to indicate the effect
    of an intervention.

    The experimental and observational data are either assumed to come to the same population,
    or from random samples of the population. If the data are from a sample, the bounds may
    be incorrectly calculated because the relevant quantities in the Tian-Pearl equations are
    defined e.g. as P(YifT), not P(YifT \mid S) where S corresponds to sample selection.
    Bareinboim and Pearl (https://www.pnas.org/doi/10.1073/pnas.1510507113) discuss conditions
    under which P(YifT) can be recovered from P(YifT \mid S).
    """

    # Probabilities calculated from observational data
    Y1 = data_obs[Y].mean()
    T1Y0 = (
        data_obs.loc[(data_obs[T] == 1) & (data_obs[Y] == 0)].shape[0]
        / data_obs.shape[0]
    )
    T1Y1 = (
        data_obs.loc[(data_obs[T] == 1) & (data_obs[Y] == 1)].shape[0]
        / data_obs.shape[0]
    )
    T0Y0 = (
        data_obs.loc[(data_obs[T] == 0) & (data_obs[Y] == 0)].shape[0]
        / data_obs.shape[0]
    )
    T0Y1 = (
        data_obs.loc[(data_obs[T] == 0) & (data_obs[Y] == 1)].shape[0]
        / data_obs.shape[0]
    )

    # Probabilities calculated from experimental data
    Y1doT1 = data_exp.loc[data_exp[T] == 1, Y].mean()
    Y1doT0 = data_exp.loc[data_exp[T] == 0, Y].mean()
    Y0doT0 = 1 - Y1doT0

    if type == "PNS":

        lb_args = [0, Y1doT1 - Y1doT0, Y1 - Y1doT0, Y1doT1 - Y1]

        ub_args = [Y1doT1, Y0doT0, T1Y1 + T0Y0, Y1doT1 - Y1doT0 + T1Y0 + T0Y1]

    if type == "PN":

        lb_args = [0, (Y1 - Y1doT0) / T1Y1]
        ub_args = [1, (Y0doT0 - T0Y0) / T1Y1]

    if type == "PS":

        lb_args = [0, (Y1doT1 - Y1) / T0Y0]
        ub_args = [1, (Y1doT1 - T1Y1) / T0Y0]

    lower_bound = max(lb_args)
    upper_bound = min(ub_args)

    return lower_bound, upper_bound
