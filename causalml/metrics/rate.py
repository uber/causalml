from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logging

plt.style.use("fivethirtyeight")
RANDOM_COL = "Random"

logger = logging.getLogger("causalml")


def get_toc(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
    """Get the Targeting Operator Characteristic (TOC) of model estimates in population.

    TOC(q) is the difference between the ATE among the top-q fraction of units ranked
    by the prioritization score and the overall ATE. A positive TOC at low q indicates
    the model successfully identifies units with above-average treatment benefit.

    If the true treatment effect is provided (e.g. in synthetic data), it's used directly
    to calculate TOC. Otherwise, it's estimated as the difference between the mean outcomes
    of the treatment and control groups in each quantile band.

    For details, see Yadlowsky et al. (2021), `Evaluating Treatment Prioritization Rules
    via Rank-Weighted Average Treatment Effects`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): deprecated

    Returns:
        (pandas.DataFrame): TOC values of model estimates in population, indexed by quantile q
    """
    assert (
        (outcome_col in df.columns and df[outcome_col].notnull().all())
        and (treatment_col in df.columns and df[treatment_col].notnull().all())
        or (
            treatment_effect_col in df.columns
            and df[treatment_effect_col].notnull().all()
        )
    ), "{outcome_col} and {treatment_col}, or {treatment_effect_col} should be present without null.".format(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
    )

    df = df.copy()

    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, treatment_effect_col]
    ]

    use_oracle = (
        treatment_effect_col in df.columns and df[treatment_effect_col].notnull().all()
    )

    if use_oracle:
        overall_ate = df[treatment_effect_col].mean()
    else:
        treated = df[treatment_col] == 1
        overall_ate = (
            df.loc[treated, outcome_col].mean() - df.loc[~treated, outcome_col].mean()
        )

    n_total = len(df)
    quantiles = np.linspace(0, 1, n_total + 1)[1:]

    toc = []
    for col in model_names:
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        toc_values = []

        for q in quantiles:
            top_k = max(1, int(np.ceil(q * n_total)))
            top_subset = sorted_df.iloc[:top_k]

            if use_oracle:
                subset_ate = top_subset[treatment_effect_col].mean()
            else:
                t_mask = top_subset[treatment_col] == 1
                c_mask = top_subset[treatment_col] == 0
                if t_mask.sum() == 0 or c_mask.sum() == 0:
                    subset_ate = overall_ate
                else:
                    subset_ate = (
                        top_subset.loc[t_mask, outcome_col].mean()
                        - top_subset.loc[c_mask, outcome_col].mean()
                    )

            toc_values.append(subset_ate - overall_ate)

        toc.append(pd.Series(toc_values, index=quantiles))

    toc = pd.concat(toc, join="inner", axis=1)
    toc.loc[0] = np.zeros((toc.shape[1],))
    toc = toc.sort_index().interpolate()
    toc.columns = model_names
    toc.index.name = "q"

    if normalize:
        toc = toc.div(np.abs(toc.iloc[-1, :]), axis=1)

    return toc


def rate_score(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    weighting="autoc",
    normalize=False,
    random_seed=42,
):
    """Calculate the Rank-weighted Average Treatment Effect (RATE) score.

    RATE is the weighted area under the Targeting Operator Characteristic (TOC) curve:

        RATE = integral_0^1 alpha(q) * TOC(q) dq

    Two standard weighting schemes are supported (Yadlowsky et al., 2021):

    - ``"autoc"``: alpha(q) = 1/q. Places more weight on the highest-priority units.
      Most powerful when treatment effects are concentrated in a small subgroup.

    - ``"qini"``: alpha(q) = q. Uniform weighting across units; reduces to the Qini
      coefficient. More powerful when treatment effects are diffuse across the population.

    A positive RATE indicates the prioritization rule effectively identifies units with
    above-average treatment benefit. A RATE near zero suggests little heterogeneity or
    a poor prioritization rule.

    For details, see Yadlowsky et al. (2021), `Evaluating Treatment Prioritization Rules
    via Rank-Weighted Average Treatment Effects`. https://arxiv.org/abs/2111.07966

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        weighting (str, optional): the weighting scheme for the RATE integral.
            One of ``"autoc"`` (default) or ``"qini"``.
        normalize (bool, optional): whether to normalize the TOC curve before scoring
        random_seed (int, optional): deprecated

    Returns:
        (pandas.Series): RATE scores of model estimates
    """
    assert weighting in (
        "autoc",
        "qini",
    ), "{} weighting is not implemented. Select one of {}".format(
        weighting, ("autoc", "qini")
    )

    toc = get_toc(
        df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
    )

    quantiles = toc.index.values  # shape (n,), 0 included

    # Use midpoints to avoid division by zero for autoc at q=0
    q_mid = (quantiles[:-1] + quantiles[1:]) / 2
    toc_mid = (toc.iloc[:-1].values + toc.iloc[1:].values) / 2

    if weighting == "autoc":
        weights = 1.0 / q_mid
    else:
        weights = q_mid

    # Normalize weights so they sum to 1 over the integration domain
    weights = weights / weights.sum()

    rate = pd.Series(
        np.average(toc_mid, axis=0, weights=weights),
        index=toc.columns,
    )
    rate.name = "RATE ({})".format(weighting)
    return rate


def plot_toc(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
    n=100,
    figsize=(8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the Targeting Operator Characteristic (TOC) curve of model estimates.

    The TOC(q) shows the excess ATE when treating only the top-q fraction of units
    prioritized by a model score, relative to the overall ATE. A positive and steeply
    decreasing curve indicates the model effectively ranks high-benefit units first.

    If the true treatment effect is provided (e.g. in synthetic data), it's used directly.
    Otherwise, it's estimated from observed outcomes and treatment assignments.

    For details, see Yadlowsky et al. (2021), `Evaluating Treatment Prioritization Rules
    via Rank-Weighted Average Treatment Effects`. https://arxiv.org/abs/2111.07966

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): deprecated
        n (int, optional): the number of samples to be used for plotting
        figsize (tuple, optional): the size of the figure to plot
        ax (plt.Axes, optional): an existing axes object to draw on

    Returns:
        (plt.Axes): the matplotlib Axes with the TOC plot
    """
    toc = get_toc(
        df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
    )

    if (n is not None) and (n < toc.shape[0]):
        toc = toc.iloc[np.linspace(0, len(toc) - 1, n, endpoint=True).astype(int)]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax = toc.plot(ax=ax)

    # Random baseline (TOC = 0 everywhere)
    ax.plot(
        [toc.index[0], toc.index[-1]],
        [0, 0],
        label=RANDOM_COL,
        color="k",
        linestyle="--",
    )
    ax.legend()
    ax.set_xlabel("Fraction treated (q)")
    ax.set_ylabel("TOC(q)")
    ax.set_title("Targeting Operator Characteristic (TOC)")

    return ax
