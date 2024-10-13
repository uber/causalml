from matplotlib import pyplot as plt
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from ..inference.meta.tmle import TMLELearner


plt.style.use("fivethirtyeight")
sns.set_palette("Paired")
RANDOM_COL = "Random"

logger = logging.getLogger("causalml")


def plot(
    df,
    kind="gain",
    tmle=False,
    n=100,
    figsize=(8, 8),
    ci=False,
    plot_chance_level=True,
    chance_level_kw=None,
    *args,
    **kwarg,
):
    """Plot one of the lift/gain/Qini charts of model estimates.

    A factory method for `plot_lift()`, `plot_gain()`, `plot_qini()`, `plot_tmlegain()` and `plot_tmleqini()`.
    For details, pleas see docstrings of each function.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns.
        kind (str, optional): the kind of plot to draw. 'lift', 'gain', and 'qini' are supported.
        n (int, optional): the number of samples to be used for plotting.
        figsize (set of float, optional): the size of the figure to plot.
        ci (bool, optional): whether to plot confidence intervals or not. Only available for `tmle=True`.
            Default is False.
        plot_chance_level (bool, optional): whether to plot the chance level (i.e., random) line or not.
            Default is True.
        chance_level_line_kw (dict, optional): the keyword arguments for the chance level line. Default is None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    if tmle:
        catalog = {"gain": get_tmlegain, "qini": get_tmleqini}
    else:
        catalog = {"lift": get_cumlift, "gain": get_cumgain, "qini": get_qini}

    assert (
        kind in catalog.keys()
    ), "{} plot is not implemented. Select one of {}".format(kind, catalog.keys())

    _, ax = plt.subplots(figsize=figsize)
    if tmle:
        df = catalog[kind](df, ci=ci, *args, **kwarg)

        if ci:
            model_names = [x.replace(" LB", "") for x in df.columns]
            model_names = list(set([x.replace(" UB", "") for x in model_names]))

            cmap = plt.get_cmap("tab10")
            cindex = 0

            for col in model_names:
                lb_col = col + " LB"
                up_col = col + " UB"

                ax.plot(df.index, df[col], color=cmap(cindex))
                ax.fill_between(
                    df.index,
                    df[lb_col],
                    df[up_col],
                    color=cmap(cindex),
                    alpha=0.25,
                )
                cindex += 1

            ax.legend()
        else:
            ax = df.plot(ax=ax)

    else:
        df = catalog[kind](df, *args, **kwarg)

        if (n is not None) and (n < df.shape[0]):
            df = df.iloc[np.linspace(0, df.index[-1], n, endpoint=True).astype(int)]

        ax = df.plot(ax=ax)

    if plot_chance_level:
        chance_level_line_kw = {
            "label": RANDOM_COL,
            "color": "k",
            "linestyle": "--",
        }

        if chance_level_kw is not None:
            chance_level_line_kw.update(**chance_level_kw)

        ax.plot([0, df.index[-1]], [0, df.iloc[-1, 0]], **chance_level_line_kw)
        ax.legend()

    ax.set_xlabel("Population")
    ax.set_ylabel("{}".format(kind.title()))


def get_cumlift(
    df, outcome_col="y", treatment_col="w", treatment_effect_col="tau", random_seed=42
):
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
        random_seed (int, optional): deprecated

    Returns:
        (pandas.DataFrame): average uplifts of model estimates in cumulative population
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

    lift = []
    for i, col in enumerate(model_names):
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1

        if treatment_effect_col in sorted_df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            lift.append(sorted_df[treatment_effect_col].cumsum() / sorted_df.index)
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            sorted_df["cumsum_tr"] = sorted_df[treatment_col].cumsum()
            sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
            sorted_df["cumsum_y_tr"] = (
                sorted_df[outcome_col] * sorted_df[treatment_col]
            ).cumsum()
            sorted_df["cumsum_y_ct"] = (
                sorted_df[outcome_col] * (1 - sorted_df[treatment_col])
            ).cumsum()

            lift.append(
                sorted_df["cumsum_y_tr"] / sorted_df["cumsum_tr"]
                - sorted_df["cumsum_y_ct"] / sorted_df["cumsum_ct"]
            )

    lift = pd.concat(lift, join="inner", axis=1)
    lift.loc[0] = np.zeros((lift.shape[1],))
    lift = lift.sort_index().interpolate()

    lift.columns = model_names

    return lift


def get_cumgain(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
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
        random_seed (int, optional): deprecated

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """

    lift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col)

    # cumulative gain = cumulative lift x (# of population)
    gain = lift.mul(lift.index.values, axis=0)

    if normalize:
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

    return gain


def get_qini(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
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
        random_seed (int, optional): deprecated

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
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

    qini = []
    for i, col in enumerate(model_names):
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1
        sorted_df["cumsum_tr"] = sorted_df[treatment_col].cumsum()

        if treatment_effect_col in sorted_df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            l = (
                sorted_df[treatment_effect_col].cumsum()
                / sorted_df.index
                * sorted_df["cumsum_tr"]
            )
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
            sorted_df["cumsum_y_tr"] = (
                sorted_df[outcome_col] * sorted_df[treatment_col]
            ).cumsum()
            sorted_df["cumsum_y_ct"] = (
                sorted_df[outcome_col] * (1 - sorted_df[treatment_col])
            ).cumsum()

            l = (
                sorted_df["cumsum_y_tr"]
                - sorted_df["cumsum_y_ct"]
                * sorted_df["cumsum_tr"]
                / sorted_df["cumsum_ct"]
            )

        qini.append(l)

    qini = pd.concat(qini, join="inner", axis=1)
    qini.loc[0] = np.zeros((qini.shape[1],))
    qini = qini.sort_index().interpolate()

    qini.columns = model_names

    if normalize:
        qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

    return qini


def get_tmlegain(
    df,
    inference_col,
    learner=LGBMRegressor(num_leaves=64, learning_rate=0.05, n_estimators=300),
    outcome_col="y",
    treatment_col="w",
    p_col="p",
    n_segment=5,
    cv=None,
    calibrate_propensity=True,
    ci=False,
):
    """Get TMLE based average uplifts of model estimates of segments.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        inferenece_col (list of str): a list of columns that used in learner for inference
        learner (optional): a model used by TMLE to estimate the outcome
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        p_col (str, optional): the column name for propensity score
        n_segment (int, optional): number of segment that TMLE will estimated for each
        cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        calibrate_propensity (bool, optional): whether calibrate propensity score or not
        ci (bool, optional): whether return confidence intervals for ATE or not
    Returns:
        (pandas.DataFrame): cumulative gains of model estimates based of TMLE
    """
    assert (
        (outcome_col in df.columns and df[outcome_col].notnull().all())
        and (treatment_col in df.columns and df[treatment_col].notnull().all())
        or (p_col in df.columns and df[p_col].notnull().all())
    ), "{outcome_col} and {treatment_col}, or {p_col} should be present without null.".format(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        p_col=p_col,
    )

    inference_col = [x for x in inference_col if x in df.columns]

    # Initialize TMLE
    tmle = TMLELearner(learner, cv=cv, calibrate_propensity=calibrate_propensity)
    ate_all, ate_all_lb, ate_all_ub = tmle.estimate_ate(
        X=df[inference_col], p=df[p_col], treatment=df[treatment_col], y=df[outcome_col]
    )

    df = df.copy()
    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, p_col] + inference_col
    ]

    lift = []
    lift_lb = []
    lift_ub = []

    for col in model_names:
        # Create `n_segment` equal segments from sorted model estimates. Rank is used to break ties.
        # ref: https://stackoverflow.com/a/46979206/3216742
        segments = pd.qcut(df[col].rank(method="first"), n_segment, labels=False)

        ate_model, ate_model_lb, ate_model_ub = tmle.estimate_ate(
            X=df[inference_col],
            p=df[p_col],
            treatment=df[treatment_col],
            y=df[outcome_col],
            segment=segments,
        )
        lift_model = [0.0] * (n_segment + 1)
        lift_model[n_segment] = ate_all[0]
        for i in range(1, n_segment):
            lift_model[i] = (
                ate_model[0][n_segment - i] * (1 / n_segment) + lift_model[i - 1]
            )
        lift.append(lift_model)

        if ci:
            lift_lb_model = [0.0] * (n_segment + 1)
            lift_lb_model[n_segment] = ate_all_lb[0]

            lift_ub_model = [0.0] * (n_segment + 1)
            lift_ub_model[n_segment] = ate_all_ub[0]
            for i in range(1, n_segment):
                lift_lb_model[i] = (
                    ate_model_lb[0][n_segment - i] * (1 / n_segment)
                    + lift_lb_model[i - 1]
                )
                lift_ub_model[i] = (
                    ate_model_ub[0][n_segment - i] * (1 / n_segment)
                    + lift_ub_model[i - 1]
                )

            lift_lb.append(lift_lb_model)
            lift_ub.append(lift_ub_model)

    lift = pd.DataFrame(lift).T
    lift.columns = model_names

    if ci:
        lift_lb = pd.DataFrame(lift_lb).T
        lift_lb.columns = [x + " LB" for x in model_names]

        lift_ub = pd.DataFrame(lift_ub).T
        lift_ub.columns = [x + " UB" for x in model_names]
        lift = pd.concat([lift, lift_lb, lift_ub], axis=1)

    lift.index = lift.index / n_segment

    return lift


def get_tmleqini(
    df,
    inference_col,
    learner=LGBMRegressor(num_leaves=64, learning_rate=0.05, n_estimators=300),
    outcome_col="y",
    treatment_col="w",
    p_col="p",
    n_segment=5,
    cv=None,
    calibrate_propensity=True,
    ci=False,
    normalize=False,
):
    """Get TMLE based Qini of model estimates by segments.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        inferenece_col (list of str): a list of columns that used in learner for inference
        learner(optional): a model used by TMLE to estimate the outcome
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        p_col (str, optional): the column name for propensity score
        n_segment (int, optional): number of segment that TMLE will estimated for each
        cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        calibrate_propensity (bool, optional): whether calibrate propensity score or not
        ci (bool, optional): whether return confidence intervals for ATE or not
    Returns:
        (pandas.DataFrame): cumulative gains of model estimates based of TMLE
    """
    assert (
        (outcome_col in df.columns and df[outcome_col].notnull().all())
        and (treatment_col in df.columns and df[treatment_col].notnull().all())
        or (p_col in df.columns and df[p_col].notnull().all())
    ), "{outcome_col} and {treatment_col}, or {p_col} should be present without null.".format(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        p_col=p_col,
    )

    inference_col = [x for x in inference_col if x in df.columns]

    # Initialize TMLE
    tmle = TMLELearner(learner, cv=cv, calibrate_propensity=calibrate_propensity)
    ate_all, ate_all_lb, ate_all_ub = tmle.estimate_ate(
        X=df[inference_col], p=df[p_col], treatment=df[treatment_col], y=df[outcome_col]
    )

    df = df.copy()
    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, p_col] + inference_col
    ]

    qini = []
    qini_lb = []
    qini_ub = []

    for col in model_names:
        # Create `n_segment` equal segments from sorted model estimates. Rank is used to break ties.
        # ref: https://stackoverflow.com/a/46979206/3216742
        segments = pd.qcut(df[col].rank(method="first"), n_segment, labels=False)

        ate_model, ate_model_lb, ate_model_ub = tmle.estimate_ate(
            X=df[inference_col],
            p=df[p_col],
            treatment=df[treatment_col],
            y=df[outcome_col],
            segment=segments,
        )

        qini_model = [0]
        for i in range(1, n_segment):
            n_tr = df[segments == (n_segment - i)][treatment_col].sum()
            qini_model.append(ate_model[0][n_segment - i] * n_tr)

        qini.append(qini_model)

        if ci:
            qini_lb_model = [0]
            qini_ub_model = [0]
            for i in range(1, n_segment):
                n_tr = df[segments == (n_segment - i)][treatment_col].sum()
                qini_lb_model.append(ate_model_lb[0][n_segment - i] * n_tr)
                qini_ub_model.append(ate_model_ub[0][n_segment - i] * n_tr)

            qini_lb.append(qini_lb_model)
            qini_ub.append(qini_ub_model)

    qini = pd.DataFrame(qini).T
    qini.columns = model_names

    if ci:
        qini_lb = pd.DataFrame(qini_lb).T
        qini_lb.columns = [x + " LB" for x in model_names]

        qini_ub = pd.DataFrame(qini_ub).T
        qini_ub.columns = [x + " UB" for x in model_names]
        qini = pd.concat([qini, qini_lb, qini_ub], axis=1)

    qini = qini.cumsum()
    qini.loc[n_segment] = ate_all[0] * df[treatment_col].sum()
    qini.index = np.linspace(0, 1, n_segment + 1) * df.shape[0]

    return qini


def plot_gain(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
    n=100,
    figsize=(8, 8),
):
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

    plot(
        df,
        kind="gain",
        n=n,
        figsize=figsize,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
    )


def plot_lift(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    random_seed=42,
    n=100,
    figsize=(8, 8),
):
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
        random_seed (int, optional): deprecated
        n (int, optional): the number of samples to be used for plotting
    """

    plot(
        df,
        kind="lift",
        n=n,
        figsize=figsize,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
    )


def plot_qini(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
    n=100,
    figsize=(8, 8),
):
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
        random_seed (int, optional): deprecated
        n (int, optional): the number of samples to be used for plotting
        ci (bool, optional): whether return confidence intervals for ATE or not
    """

    plot(
        df,
        kind="qini",
        n=n,
        figsize=figsize,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
    )


def plot_tmlegain(
    df,
    inference_col,
    learner=LGBMRegressor(
        num_leaves=64, learning_rate=0.05, n_estimators=300, verbose=-1
    ),
    outcome_col="y",
    treatment_col="w",
    p_col="tau",
    n_segment=5,
    cv=None,
    calibrate_propensity=True,
    ci=False,
    figsize=(8, 8),
):
    """Plot the lift chart based of TMLE estimation

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        inferenece_col (list of str): a list of columns that used in learner for inference
        learner (optional): a model used by TMLE to estimate the outcome
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        p_col (str, optional): the column name for propensity score
        n_segment (int, optional): number of segment that TMLE will estimated for each
        cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        calibrate_propensity (bool, optional): whether calibrate propensity score or not
        ci (bool, optional): whether return confidence intervals for ATE or not
    """

    plot(
        df,
        kind="gain",
        tmle=True,
        figsize=figsize,
        ci=ci,
        learner=learner,
        inference_col=inference_col,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        p_col=p_col,
        n_segment=n_segment,
        cv=cv,
        calibrate_propensity=calibrate_propensity,
    )


def plot_tmleqini(
    df,
    inference_col,
    learner=LGBMRegressor(num_leaves=64, learning_rate=0.05, n_estimators=300),
    outcome_col="y",
    treatment_col="w",
    p_col="tau",
    n_segment=5,
    cv=None,
    calibrate_propensity=True,
    ci=False,
    figsize=(8, 8),
):
    """Plot the qini chart based of TMLE estimation

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        inferenece_col (list of str): a list of columns that used in learner for inference
        learner (optional): a model used by TMLE to estimate the outcome
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        p_col (str, optional): the column name for propensity score
        n_segment (int, optional): number of segment that TMLE will estimated for each
        cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        calibrate_propensity (bool, optional): whether calibrate propensity score or not
        ci (bool, optional): whether return confidence intervals for ATE or not
    """

    plot(
        df,
        kind="qini",
        tmle=True,
        figsize=figsize,
        ci=ci,
        learner=learner,
        inference_col=inference_col,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        p_col=p_col,
        n_segment=n_segment,
        cv=cv,
        calibrate_propensity=calibrate_propensity,
    )


def auuc_score(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=True,
    tmle=False,
    *args,
    **kwarg,
):
    """Calculate the AUUC (Area Under the Uplift Curve) score.

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the AUUC score
    """

    if not tmle:
        cumgain = get_cumgain(
            df, outcome_col, treatment_col, treatment_effect_col, normalize
        )
    else:
        cumgain = get_tmlegain(
            df, outcome_col=outcome_col, treatment_col=treatment_col, *args, **kwarg
        )
    return cumgain.sum() / cumgain.shape[0]


def qini_score(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=True,
    tmle=False,
    *args,
    **kwarg,
):
    """Calculate the Qini score: the area between the Qini curves of a model and random.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the Qini score
    """

    if not tmle:
        qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize)
    else:
        qini = get_tmleqini(
            df, outcome_col=outcome_col, treatment_col=treatment_col, *args, **kwarg
        )

    random_area = np.linspace(qini.iloc[0, 0], qini.iloc[-1, 0], qini.shape[0]).sum()
    return (qini.sum(axis=0) - random_area) / qini.shape[0]


def plot_ps_diagnostics(df, covariate_col, treatment_col="w", p_col="p", bal_tol=0.1):
    """Plot covariate balances (standardized differences between the treatment and the control)
    before and after weighting the sample using the inverse probability of treatment weights.

     Args:
        df (pandas.DataFrame): a data frame containing the covariates and treatment indicator
        covariate_col (list of str): a list of columns that are used a covariates
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        p_col (str, optional): the column name for propensity score
    """
    X = df[covariate_col]
    W = df[treatment_col]
    PS = df[p_col]

    IPTW = get_simple_iptw(W, PS)

    diffs_pre = get_std_diffs(X, W, weighted=False)
    num_unbal_pre = (np.abs(diffs_pre) > bal_tol).sum()[0]

    diffs_post = get_std_diffs(X, W, IPTW, weighted=True)
    num_unbal_post = (np.abs(diffs_post) > bal_tol).sum()[0]

    diff_plot = _plot_std_diffs(
        diffs_pre, num_unbal_pre, diffs_post, num_unbal_post, bal_tol=bal_tol
    )

    return diff_plot


def _plot_std_diffs(diffs_pre, num_unbal_pre, diffs_post, num_unbal_post, bal_tol=0.1):
    fig, ax1 = plt.subplots()

    color = "#EA2566"

    sds_pre = pd.DataFrame(
        {"std_diff": diffs_pre[0], "covariate": diffs_pre.index, "prepost": "pre"}
    )
    sds_post = pd.DataFrame(
        {"std_diff": diffs_post[0], "covariate": diffs_post.index, "prepost": "post"}
    )

    sds = pd.concat([sds_pre, sds_post], ignore_index=True)

    sns.stripplot(data=sds, x="std_diff", y="covariate", hue="prepost", ax=ax1)

    ax1.set_xlabel(
        "Pre/Post Number of unbalanced covariates: {num_unbal_pre}/{num_unbal_post}".format(
            num_unbal_pre=num_unbal_pre, num_unbal_post=num_unbal_post
        ),
        fontsize=14,
    )
    ax1.axvline(x=-bal_tol, ymin=0, ymax=1, color=color, linestyle="--", lw=2)
    ax1.axvline(x=bal_tol, ymin=0, ymax=1, color=color, linestyle="--", lw=2)

    fig.suptitle("Standardized differences in means", fontsize=16)

    return fig


def get_simple_iptw(W, propensity_score):
    IPTW = (W / propensity_score) + (1 - W) / (1 - propensity_score)

    return IPTW


def get_std_diffs(X, W, weight=None, weighted=False, numeric_threshold=5):
    """Calculate the inverse probability of treatment weighted standardized
    differences in covariate means between the treatment and the control.
    If weighting is set to 'False', calculate unweighted standardized
    differences. Accepts only continuous and binary numerical variables.
    """
    cont_cols, prop_cols = _get_numeric_vars(X, threshold=numeric_threshold)
    cols = cont_cols + prop_cols

    if len(cols) == 0:
        raise ValueError(
            "No variable passed the test for continuous or binary variables."
        )

    treat = W == 1
    contr = W == 0

    X_1 = X.loc[treat, cols]
    X_0 = X.loc[contr, cols]

    cont_index = np.array([col in cont_cols for col in cols])
    prop_index = np.array([col in prop_cols for col in cols])

    std_diffs_cont = np.empty(sum(cont_index))
    std_diffs_prop = np.empty(sum(prop_index))

    if weighted:
        assert (
            weight is not None
        ), 'weight should be provided when weighting is set to "True"'

        weight_1 = weight[treat]
        weight_0 = weight[contr]

        X_1_mean, X_1_var = np.apply_along_axis(
            lambda x: _get_wmean_wvar(x, weight_1), 0, X_1
        )
        X_0_mean, X_0_var = np.apply_along_axis(
            lambda x: _get_wmean_wvar(x, weight_0), 0, X_0
        )

    elif not weighted:
        X_1_mean, X_1_var = np.apply_along_axis(lambda x: _get_mean_var(x), 0, X_1)
        X_0_mean, X_0_var = np.apply_along_axis(lambda x: _get_mean_var(x), 0, X_0)

    X_1_mean_cont, X_1_var_cont = X_1_mean[cont_index], X_1_var[cont_index]
    X_0_mean_cont, X_0_var_cont = X_0_mean[cont_index], X_0_var[cont_index]

    std_diffs_cont = (X_1_mean_cont - X_0_mean_cont) / np.sqrt(
        (X_1_var_cont + X_0_var_cont) / 2
    )

    X_1_mean_prop = X_1_mean[prop_index]
    X_0_mean_prop = X_0_mean[prop_index]

    std_diffs_prop = (X_1_mean_prop - X_0_mean_prop) / np.sqrt(
        ((X_1_mean_prop * (1 - X_1_mean_prop)) + (X_0_mean_prop * (1 - X_0_mean_prop)))
        / 2
    )

    std_diffs = np.concatenate([std_diffs_cont, std_diffs_prop], axis=0)
    std_diffs_df = pd.DataFrame(std_diffs, index=cols)

    return std_diffs_df


def _get_numeric_vars(X, threshold=5):
    """Attempt to determine which variables are numeric and which
    are categorical. The threshold for a 'continuous' variable
    is set to 5 by default.
    """

    cont = [
        (not hasattr(X.iloc[:, i], "cat")) and (X.iloc[:, i].nunique() >= threshold)
        for i in range(X.shape[1])
    ]

    prop = [X.iloc[:, i].nunique() == 2 for i in range(X.shape[1])]

    cont_cols = list(X.loc[:, cont].columns)
    prop_cols = list(X.loc[:, prop].columns)

    dropped = set(X.columns) - set(cont_cols + prop_cols)

    if dropped:
        logger.info(
            'Some non-binary variables were dropped because they had fewer than {} unique values or were of the \
                     dtype "cat". The dropped variables are: {}'.format(
                threshold, dropped
            )
        )

    return cont_cols, prop_cols


def _get_mean_var(X):
    """Calculate the mean and variance of a variable."""
    mean = X.mean()
    var = X.var()

    return [mean, var]


def _get_wmean_wvar(X, weight):
    """
    Calculate the weighted mean of a variable given an arbitrary
    sample weight. Formulas from:

    Austin, Peter C., and Elizabeth A. Stuart. 2015. Moving towards Best
    Practice When Using Inverse Probability of Treatment Weighting (IPTW)
    Using the Propensity Score to Estimate Causal Treatment Effects in
    Observational Studies.
    Statistics in Medicine 34 (28): 3661 79. https://doi.org/10.1002/sim.6607.
    """
    weighted_mean = np.sum(weight * X) / np.sum(weight)
    weighted_var = (
        np.sum(weight) / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2)))
    ) * (np.sum(weight * np.power((X - weighted_mean), 2)))

    return [weighted_mean, weighted_var]
