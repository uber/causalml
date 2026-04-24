from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import logging

plt.style.use("fivethirtyeight")
RANDOM_COL = "Random"

logger = logging.getLogger("causalml")


def _compute_rate_from_toc(toc, weighting):
    """Compute the RATE scalar from a TOC DataFrame.

    Args:
        toc (pandas.DataFrame): TOC curve indexed by quantile q
        weighting (str): one of ``"autoc"`` or ``"qini"``

    Returns:
        (pandas.Series): RATE score for each model column
    """
    quantiles = toc.index.values
    q_mid = (quantiles[:-1] + quantiles[1:]) / 2
    toc_mid = (toc.iloc[:-1].values + toc.iloc[1:].values) / 2
    if weighting == "autoc":
        weights = 1.0 / q_mid
    else:
        weights = q_mid
    weights = weights / weights.sum()
    return pd.Series(
        np.average(toc_mid, axis=0, weights=weights),
        index=toc.columns,
    )


def get_toc(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
):
    """Get the Targeting Operator Characteristic (TOC) of model estimates in population.

    TOC(q) is the difference between the ATE among the top-q fraction of units ranked
    by the prioritization score and the overall ATE. A positive TOC at low q indicates
    the model successfully identifies units with above-average treatment benefit.

    By definition, TOC(0) = 0 and TOC(1) = 0 (the subset ATE equals the overall ATE
    when the entire population is selected).

    If the true treatment effect is provided (e.g. in synthetic data), it's used directly
    to calculate TOC. Otherwise, it's estimated as the difference between the mean outcomes
    of the treatment and control groups in each quantile band.

    Note: when using observed outcomes, if a quantile band contains only treated or only
    control units, the code falls back to TOC(q) = 0 for that band (i.e., subset ATE is
    set to the overall ATE). This is a conservative approximation and is logged as a warning.

    For details, see Yadlowsky et al. (2021), `Evaluating Treatment Prioritization Rules
    via Rank-Weighted Average Treatment Effects`. https://arxiv.org/abs/2111.07966

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the TOC curve by its maximum
            absolute value. Uses max(|TOC|) as the reference to avoid division by zero
            at q=1 where TOC is always zero by definition.

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

    toc = []
    for col in model_names:
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)

        if use_oracle:
            # O(n) via cumulative sum
            cumsum_tau = sorted_df[treatment_effect_col].cumsum().values
            counts = np.arange(1, n_total + 1)
            subset_ates = cumsum_tau / counts
        else:
            cumsum_tr = sorted_df[treatment_col].cumsum().values
            cumsum_ct = np.arange(1, n_total + 1) - cumsum_tr
            cumsum_y_tr = (
                (sorted_df[outcome_col] * sorted_df[treatment_col]).cumsum().values
            )
            cumsum_y_ct = (
                (sorted_df[outcome_col] * (1 - sorted_df[treatment_col]))
                .cumsum()
                .values
            )

            # Guard against division by zero when a band is all-treated or all-control;
            # fall back to overall_ate (TOC = 0) for those positions.
            with np.errstate(invalid="ignore", divide="ignore"):
                subset_ates = np.where(
                    (cumsum_tr == 0) | (cumsum_ct == 0),
                    overall_ate,
                    cumsum_y_tr / cumsum_tr - cumsum_y_ct / cumsum_ct,
                )

            if np.any((cumsum_tr == 0) | (cumsum_ct == 0)):
                logger.warning(
                    "Some quantile bands contain only treated or only control units "
                    "for column '%s'. TOC is set to 0 for those positions.",
                    col,
                )

        toc_values = subset_ates - overall_ate
        toc.append(pd.Series(toc_values, index=np.linspace(0, 1, n_total + 1)[1:]))

    toc = pd.concat(toc, join="inner", axis=1)
    toc.loc[0] = np.zeros((toc.shape[1],))
    toc = toc.sort_index().interpolate()
    toc.columns = model_names
    toc.index.name = "q"

    if normalize:
        # Normalize by max absolute value rather than the value at q=1, which is
        # always zero by definition and would cause division by zero.
        max_abs = toc.abs().max()
        max_abs = max_abs.replace(0, 1)  # guard for flat TOC curves
        toc = toc.div(max_abs, axis=1)

    return toc


def rate_score(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    weighting="autoc",
    normalize=False,
    return_ci=False,
    n_bootstrap=200,
    alpha=0.05,
    random_state=None,
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

    Note: the integral is approximated via a weighted mean over the discrete quantile grid
    using midpoint values. Weights are normalized to sum to 1 (i.e. ``weights / weights.sum()``),
    so the absolute scale matches the TOC values but may differ slightly from the paper's
    continuous integral definition. Model rankings are preserved.

    When return_ci=True, standard errors and confidence intervals are estimated via the
    half-sample bootstrap (m = n // 2 draws without replacement), which gives valid
    coverage for the RATE functional per the Yadlowsky et al. (2021) functional CLT.
    The p-value tests H0: RATE = 0 (i.e. the model's prioritization is no better than
    random) using a two-sided z-test.

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
        return_ci (bool, optional): whether to return bootstrap confidence intervals and
            p-values. Default False.
        n_bootstrap (int, optional): number of half-sample bootstrap iterations.
            Only used when return_ci=True. Default 200.
        alpha (float, optional): significance level for confidence intervals.
            Only used when return_ci=True. Default 0.05.
        random_state (int or None, optional): random seed for the bootstrap sampler.
            Pass an integer for reproducible results. Default None.

    Returns:
        If return_ci=False:
            (pandas.Series): RATE scores of model estimates
        If return_ci=True:
            (pandas.DataFrame): RATE score, standard error, CI lower bound, CI upper bound,
                and p-value for each model estimate column
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

    rate = _compute_rate_from_toc(toc, weighting)
    rate.name = "RATE ({})".format(weighting)

    if not return_ci:
        return rate

    # Half-sample bootstrap for SE and p-value
    n = len(df)
    m = n // 2
    model_names = toc.columns.tolist()
    boot_scores = {model: [] for model in model_names}

    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        df_boot = df.iloc[idx].reset_index(drop=True)
        toc_boot = get_toc(
            df_boot,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            treatment_effect_col=treatment_effect_col,
            normalize=normalize,
        )
        rate_boot = _compute_rate_from_toc(toc_boot, weighting)
        for model in model_names:
            boot_scores[model].append(rate_boot[model])

    z_crit = stats.norm.ppf(1 - alpha / 2)
    results = []
    for model in model_names:
        point = rate[model]
        boot = np.array(boot_scores[model])
        se = np.std(boot, ddof=1)
        ci_lower = point - z_crit * se
        ci_upper = point + z_crit * se
        z_stat = point / se if se > 0 else np.inf
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        results.append(
            {
                "model": model,
                "rate": point,
                "se": se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(results).set_index("model")


def plot_toc(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
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
        normalize (bool, optional): whether to normalize the TOC curve by its maximum
            absolute value before plotting
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
