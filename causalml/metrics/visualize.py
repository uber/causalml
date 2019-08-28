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
    """Get average uplifts of model estimates in cumulative quantiles.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative quantiles.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative quantiles.

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
        (pandas.DataFrame): average uplifts of model estimates in cumulative quantiles
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
        df['quantile'] = ((df.index.values * steps) // df.shape[0] + 1) * 100 / steps

        if treatment_effect_col in df.columns:
            # When treatment_effect_col is given, use it to calculate treatment effect
            # in each quantile.
            lift.append(df.groupby('quantile')[treatment_effect_col].mean())
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate treatment_effect in each quantile.
            treatment = df.loc[df[treatment_col] == 1].drop(treatment_col, axis=1)
            control = df.loc[df[treatment_col] == 0].drop(treatment_col, axis=1)

            lift.append(treatment.groupby('quantile')[outcome_col].mean() -
                        control.groupby('quantile')[outcome_col].mean())

    lift = pd.concat(lift, axis=1)
    assert lift.isna().any().any() == 0, 'NA in some quantiles. Try smaller steps.'

    lift.columns = model_names
    lift['Random'] = lift[random_cols].mean(axis=1)
    lift.drop(random_cols, axis=1, inplace=True)

    cumlift = lift.cumsum().div(lift.index.values, axis=0) * 100 / steps
    cumlift.loc[0] = [0] * cumlift.shape[1]
    cumlift.sort_index(inplace=True)

    return cumlift


def get_cumgain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                steps=100, normalize=False, random_seed=42):
    """Get cumulative gains of model estimates in quantiles.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each quantiles.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each quantiles.

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
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in quantiles
    """

    cumlift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, steps, random_seed)

    # cumulative gain = cumulative lift x (# of population)
    cumgain = cumlift.mul(cumlift.index.values, axis=0) / 100 * df.shape[0]

    if normalize:
        cumgain = cumgain.div(cumgain.iloc[-1, :], axis=1)

    return cumgain


def plot_gain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              steps=100, normalize=False, random_seed=42, figsize=(8, 8)):
    """Plot the cumulative gain chart (or uplift curve) of model estimates.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each quantiles.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each quantiles.

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
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()
    """

    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, steps, normalize, random_seed)

    cumgain.plot(figsize=figsize)
    plt.xlabel('Fraction of Population')
    plt.ylabel('Cumulative Gain')


def plot_lift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              steps=100, random_seed=42, figsize=(8, 8)):
    """Plot the lift chart of model estimates in cumulative quantiles.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative quantiles.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative quantiles.

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
    """

    cumlift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, steps, random_seed)

    cumlift.plot(figsize=figsize)
    plt.xlabel('Fraction of Population')
    plt.ylabel('Cumulative Uplift')


def auuc_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
               steps=100, normalize=True):
    """Calculate the AUUC (Area Under the Uplift Curve) score.

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        steps (int, optional): the number of quantiles
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the AUUC score
    """
    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, steps, normalize)
    return cumgain.sum() / steps
