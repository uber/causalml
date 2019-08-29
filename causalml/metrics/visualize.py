from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.style.use('fivethirtyeight')
sns.set_palette("Paired")


def get_cumlift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                steps=100, random_seed=42):
    """Get average uplifts of model estimates in cumulative population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        steps (int, optional): the number of quantiles
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): average uplifts of model estimates in cumulative population
    """

    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns)

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [x for x in df.columns if x not in [outcome_col, treatment_col,
                                                      treatment_effect_col]]

    lift = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1

        if treatment_effect_col in df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            lift.append(df[treatment_effect_col].cumsum() / df.index)
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            df['cumsum_tr'] = df[treatment_col].cumsum()
            df['cumsum_ct'] = df.index.values - df['cumsum_tr']
            df['cumsum_y_tr'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_ct'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()
            df = df.loc[(df['cumsum_tr'] > 0) & (df['cumsum_ct'] > 0)]

            lift.append(df['cumsum_y_tr'] / df['cumsum_tr'] - df['cumsum_y_ct'] / df['cumsum_ct'])

    lift = pd.concat(lift, join='inner', axis=1)

    lift.columns = model_names
    lift['Random'] = lift[random_cols].mean(axis=1)
    lift.drop(random_cols, axis=1, inplace=True)

    return lift


def get_cumgain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                normalize=False, random_seed=42):
    """Get cumulative gains of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """

    lift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, random_seed)

    # cumulative gain = cumulative lift x (# of population)
    gain = lift.mul(lift.index.values, axis=0)

    if normalize:
        gain = gain.div(gain.iloc[-1, :], axis=1)

    return gain


def get_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
             normalize=False, random_seed=42):
    """Get Qini of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """
    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns)

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [x for x in df.columns if x not in [outcome_col, treatment_col,
                                                      treatment_effect_col]]

    qini = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df['cumsum_tr'] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            l = df[treatment_effect_col].cumsum() / df.index * df['cumsum_tr']
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            df['cumsum_ct'] = df.index.values - df['cumsum_tr']
            df['cumsum_y_tr'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_ct'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()
            df = df.loc[(df['cumsum_tr'] > 0) & (df['cumsum_ct'] > 0)]

            l = df['cumsum_y_tr'] - df['cumsum_y_ct'] * df['cumsum_tr'] / df['cumsum_ct']

        qini.append(l)

    qini = pd.concat(qini, join='inner', axis=1)
    qini.columns = model_names
    qini['Random'] = qini[random_cols].mean(axis=1)
    qini.drop(random_cols, axis=1, inplace=True)

    if normalize:
        qini = qini.div(qini.iloc[-1, :], axis=1)

    return qini


def plot_gain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8)):
    """Plot the cumulative gain chart (or uplift curve) of model estimates.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()
        n (int, optional): the number of samples to be used for plotting
    """

    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, normalize, random_seed)

    if (n is None) or (n > cumgain.shape[0]):
        cumgain.plot(figsize=figsize)
    else:
        cumgain.sample(n=n, random_state=random_seed).sort_index().plot(figsize=figsize)

    plt.xlabel('Population')
    plt.ylabel('Cumulative Gain')


def plot_lift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              random_seed=42, n=100, figsize=(8, 8)):
    """Plot the lift chart of model estimates in cumulative population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        random_seed (int, optional): random seed for numpy.random.rand()
        n (int, optional): the number of samples to be used for plotting
    """

    cumlift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, random_seed)

    if (n is None) or (n > cumlift.shape[0]):
        cumlift.plot(figsize=figsize)
    else:
        cumlift.sample(n=n, random_state=random_seed).sort_index().plot(figsize=figsize)

    plt.xlabel('Population')
    plt.ylabel('Cumulative Uplift')


def plot_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8)):
    """Plot the Qini chart (or uplift curve) of model estimates.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()
        n (int, optional): the number of samples to be used for plotting
    """

    qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize, random_seed)

    if (n is None) or (n > qini.shape[0]):
        qini.plot(figsize=figsize)
    else:
        qini.sample(n=n, random_state=random_seed).sort_index().plot(figsize=figsize)

    plt.xlabel('Population')
    plt.ylabel('Cumulative Gain')


def auuc_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True):
    """Calculate the AUUC (Area Under the Uplift Curve) score.

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the AAUC score
    """
    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, normalize)
    return cumgain.sum() / cumgain.shape[0]
