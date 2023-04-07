from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from scipy.stats import entropy
import warnings

from causalml.inference.meta import (
    BaseXRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
)
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor
from causalml.propensity import ElasticNetPropensityModel
from causalml.metrics import plot_gain, get_cumgain


plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")

KEY_GENERATED_DATA = "generated_data"
KEY_ACTUAL = "Actuals"

RANDOM_SEED = 42


def get_synthetic_preds(synthetic_data_func, n=1000, estimators={}):
    """Generate predictions for synthetic data using specified function (single simulation)

    Args:
        synthetic_data_func (function): synthetic data generation function
        n (int, optional): number of samples
        estimators (dict of object): dict of names and objects of treatment effect estimators

    Returns:
        (dict): dict of the actual and estimates of treatment effects
    """
    y, X, w, tau, b, e = synthetic_data_func(n=n)

    preds_dict = {}
    preds_dict[KEY_ACTUAL] = tau
    preds_dict[KEY_GENERATED_DATA] = {
        "y": y,
        "X": X,
        "w": w,
        "tau": tau,
        "b": b,
        "e": e,
    }

    # Predict p_hat because e would not be directly observed in real-life
    p_model = ElasticNetPropensityModel()
    p_hat = p_model.fit_predict(X, w)

    if estimators:
        for name, learner in estimators.items():
            try:
                preds_dict[name] = learner.fit_predict(
                    X=X, treatment=w, y=y, p=p_hat
                ).flatten()
            except TypeError:
                preds_dict[name] = learner.fit_predict(X=X, treatment=w, y=y).flatten()
    else:
        for base_learner, label_l in zip(
            [BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor],
            ["S", "T", "X", "R"],
        ):
            for model, label_m in zip([LinearRegression, XGBRegressor], ["LR", "XGB"]):
                learner = base_learner(model())
                model_name = "{} Learner ({})".format(label_l, label_m)
                try:
                    preds_dict[model_name] = learner.fit_predict(
                        X=X, treatment=w, y=y, p=p_hat
                    ).flatten()
                except TypeError:
                    preds_dict[model_name] = learner.fit_predict(
                        X=X, treatment=w, y=y
                    ).flatten()

        learner = CausalTreeRegressor(random_state=RANDOM_SEED)
        preds_dict["Causal Tree"] = learner.fit_predict(X=X, treatment=w, y=y).flatten()

    return preds_dict


def get_synthetic_summary(synthetic_data_func, n=1000, k=1, estimators={}):
    """Generate a summary for predictions on synthetic data using specified function

    Args:
        synthetic_data_func (function): synthetic data generation function
        n (int, optional): number of samples per simulation
        k (int, optional): number of simulations
    """
    summaries = []

    for i in range(k):
        synthetic_preds = get_synthetic_preds(
            synthetic_data_func, n=n, estimators=estimators
        )
        actuals = synthetic_preds[KEY_ACTUAL]
        synthetic_summary = pd.DataFrame(
            {
                label: [preds.mean(), mse(preds, actuals)]
                for label, preds in synthetic_preds.items()
                if label != KEY_GENERATED_DATA
            },
            index=["ATE", "MSE"],
        ).T

        synthetic_summary["Abs % Error of ATE"] = np.abs(
            (synthetic_summary["ATE"] / synthetic_summary.loc[KEY_ACTUAL, "ATE"]) - 1
        )

        for label in synthetic_summary.index:
            stacked_values = np.hstack((synthetic_preds[label], actuals))
            stacked_low = np.percentile(stacked_values, 0.1)
            stacked_high = np.percentile(stacked_values, 99.9)
            bins = np.linspace(stacked_low, stacked_high, 100)

            distr = np.histogram(synthetic_preds[label], bins=bins)[0]
            distr = np.clip(distr / distr.sum(), 0.001, 0.999)
            true_distr = np.histogram(actuals, bins=bins)[0]
            true_distr = np.clip(true_distr / true_distr.sum(), 0.001, 0.999)

            kl = entropy(distr, true_distr)
            synthetic_summary.loc[label, "KL Divergence"] = kl

        summaries.append(synthetic_summary)

    summary = sum(summaries) / k
    return summary[["Abs % Error of ATE", "MSE", "KL Divergence"]]


def scatter_plot_summary(synthetic_summary, k, drop_learners=[], drop_cols=[]):
    """Generates a scatter plot comparing learner performance. Each learner's performance is plotted as a point in the
    (Abs % Error of ATE, MSE) space.

    Args:
        synthetic_summary (pd.DataFrame): summary generated by get_synthetic_summary()
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
    """
    plot_data = synthetic_summary.drop(drop_learners).drop(drop_cols, axis=1)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    xs = plot_data["Abs % Error of ATE"]
    ys = plot_data["MSE"]

    ax.scatter(xs, ys)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    for i, txt in enumerate(plot_data.index):
        ax.annotate(
            txt,
            (
                xs[i] - np.random.binomial(1, 0.5) * xlim[1] * 0.04,
                ys[i] - ylim[1] * 0.03,
            ),
        )

    ax.set_xlabel("Abs % Error of ATE")
    ax.set_ylabel("MSE")
    ax.set_title("Learner Performance (averaged over k={} simulations)".format(k))


def bar_plot_summary(
    synthetic_summary,
    k,
    drop_learners=[],
    drop_cols=[],
    sort_cols=["MSE", "Abs % Error of ATE"],
):
    """Generates a bar plot comparing learner performance.

    Args:
        synthetic_summary (pd.DataFrame): summary generated by get_synthetic_summary()
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
        sort_cols (list, optional): list of metrics (str) to sort on when plotting
    """
    plot_data = synthetic_summary.sort_values(sort_cols, ascending=True)
    plot_data = plot_data.drop(drop_learners + [KEY_ACTUAL]).drop(drop_cols, axis=1)

    plot_data.plot(kind="bar", figsize=(12, 8))
    plt.xticks(rotation=30)
    plt.title("Learner Performance (averaged over k={} simulations)".format(k))


def distr_plot_single_sim(
    synthetic_preds,
    kind="kde",
    drop_learners=[],
    bins=50,
    histtype="step",
    alpha=1,
    linewidth=1,
    bw_method=1,
):
    """Plots the distribution of each learner's predictions (for a single simulation).
    Kernel Density Estimation (kde) and actual histogram plots supported.

    Args:
        synthetic_preds (dict): dictionary of predictions generated by get_synthetic_preds()
        kind (str, optional): 'kde' or 'hist'
        drop_learners (list, optional): list of learners (str) to omit when plotting
        bins (int, optional): number of bins to plot if kind set to 'hist'
        histtype (str, optional): histogram type if kind set to 'hist'
        alpha (float, optional): alpha (transparency) for plotting
        linewidth (int, optional): line width for plotting
        bw_method (float, optional): parameter for kde
    """
    preds_for_plot = synthetic_preds.copy()

    # deleted generated data and assign actual value
    del preds_for_plot[KEY_GENERATED_DATA]
    global_lower = np.percentile(np.hstack(preds_for_plot.values()), 1)
    global_upper = np.percentile(np.hstack(preds_for_plot.values()), 99)
    learners = list(preds_for_plot.keys())
    learners = [learner for learner in learners if learner not in drop_learners]

    # Plotting
    plt.figure(figsize=(12, 8))
    colors = [
        "black",
        "red",
        "blue",
        "green",
        "cyan",
        "brown",
        "grey",
        "pink",
        "orange",
        "yellow",
    ]
    for i, (k, v) in enumerate(preds_for_plot.items()):
        if k in learners:
            if kind == "kde":
                v = pd.Series(v.flatten())
                v = v[v.between(global_lower, global_upper)]
                v.plot(
                    kind="kde",
                    bw_method=bw_method,
                    label=k,
                    linewidth=linewidth,
                    color=colors[i],
                )
            elif kind == "hist":
                plt.hist(
                    v,
                    bins=np.linspace(global_lower, global_upper, bins),
                    label=k,
                    histtype=histtype,
                    alpha=alpha,
                    linewidth=linewidth,
                    color=colors[i],
                )
            else:
                pass

    plt.xlim(global_lower, global_upper)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Distribution from a Single Simulation")


def scatter_plot_single_sim(synthetic_preds):
    """Creates a grid of scatter plots comparing each learner's predictions with the truth (for a single simulation).

    Args:
        synthetic_preds (dict): dictionary of predictions generated by get_synthetic_preds() or
            get_synthetic_preds_holdout()
    """
    preds_for_plot = synthetic_preds.copy()

    # deleted generated data and get actual column name
    del preds_for_plot[KEY_GENERATED_DATA]
    n_row = int(np.ceil(len(preds_for_plot.keys()) / 3))

    fig, axes = plt.subplots(n_row, 3, figsize=(5 * n_row, 15))
    axes = np.ravel(axes)

    for i, (label, preds) in enumerate(preds_for_plot.items()):
        axes[i].scatter(preds_for_plot[KEY_ACTUAL], preds, s=2, label="Predictions")
        axes[i].set_title(label, size=12)
        axes[i].set_xlabel("Actual", size=10)
        axes[i].set_ylabel("Prediction", size=10)
        xlim = axes[i].get_xlim()
        ylim = axes[i].get_xlim()
        axes[i].plot(
            [xlim[0], xlim[1]],
            [ylim[0], ylim[1]],
            label="Perfect Model",
            linewidth=1,
            color="grey",
        )
        axes[i].legend(loc=2, prop={"size": 10})


def get_synthetic_preds_holdout(
    synthetic_data_func, n=1000, valid_size=0.2, estimators={}
):
    """Generate predictions for synthetic data using specified function (single simulation) for train and holdout

    Args:
        synthetic_data_func (function): synthetic data generation function
        n (int, optional): number of samples
        valid_size(float,optional): validaiton/hold out data size
        estimators (dict of object): dict of names and objects of treatment effect estimators

    Returns:
        (tuple): synthetic training and validation data dictionaries:

          - preds_dict_train (dict): synthetic training data dictionary
          - preds_dict_valid (dict): synthetic validation data dictionary
    """
    y, X, w, tau, b, e = synthetic_data_func(n=n)

    (
        X_train,
        X_val,
        y_train,
        y_val,
        w_train,
        w_val,
        tau_train,
        tau_val,
        b_train,
        b_val,
        e_train,
        e_val,
    ) = train_test_split(
        X, y, w, tau, b, e, test_size=valid_size, random_state=RANDOM_SEED, shuffle=True
    )

    preds_dict_train = {}
    preds_dict_valid = {}

    preds_dict_train[KEY_ACTUAL] = tau_train
    preds_dict_valid[KEY_ACTUAL] = tau_val

    preds_dict_train["generated_data"] = {
        "y": y_train,
        "X": X_train,
        "w": w_train,
        "tau": tau_train,
        "b": b_train,
        "e": e_train,
    }
    preds_dict_valid["generated_data"] = {
        "y": y_val,
        "X": X_val,
        "w": w_val,
        "tau": tau_val,
        "b": b_val,
        "e": e_val,
    }

    # Predict p_hat because e would not be directly observed in real-life
    p_model = ElasticNetPropensityModel()
    p_hat_train = p_model.fit_predict(X_train, w_train)
    p_hat_val = p_model.fit_predict(X_val, w_val)

    for base_learner, label_l in zip(
        [BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor],
        ["S", "T", "X", "R"],
    ):
        for model, label_m in zip([LinearRegression, XGBRegressor], ["LR", "XGB"]):
            # RLearner will need to fit on the p_hat
            if label_l != "R":
                learner = base_learner(model())
                # fit the model on training data only
                learner.fit(X=X_train, treatment=w_train, y=y_train)
                try:
                    preds_dict_train[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(X=X_train, p=p_hat_train).flatten()
                    preds_dict_valid[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(X=X_val, p=p_hat_val).flatten()
                except TypeError:
                    preds_dict_train[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(
                        X=X_train, treatment=w_train, y=y_train
                    ).flatten()
                    preds_dict_valid[
                        "{} Learner ({})".format(label_l, label_m)
                    ] = learner.predict(X=X_val, treatment=w_val, y=y_val).flatten()
            else:
                learner = base_learner(model())
                learner.fit(X=X_train, p=p_hat_train, treatment=w_train, y=y_train)
                preds_dict_train[
                    "{} Learner ({})".format(label_l, label_m)
                ] = learner.predict(X=X_train).flatten()
                preds_dict_valid[
                    "{} Learner ({})".format(label_l, label_m)
                ] = learner.predict(X=X_val).flatten()

    return preds_dict_train, preds_dict_valid


def get_synthetic_summary_holdout(synthetic_data_func, n=1000, valid_size=0.2, k=1):
    """Generate a summary for predictions on synthetic data for train and holdout using specified function

    Args:
        synthetic_data_func (function): synthetic data generation function
        n (int, optional): number of samples per simulation
        valid_size(float,optional): validation/hold out data size
        k (int, optional): number of simulations


    Returns:
        (tuple): summary evaluation metrics of predictions for train and validation:

          - summary_train (pandas.DataFrame): training data evaluation summary
          - summary_train (pandas.DataFrame): validation data evaluation summary
    """

    summaries_train = []
    summaries_validation = []

    for i in range(k):
        preds_dict_train, preds_dict_valid = get_synthetic_preds_holdout(
            synthetic_data_func, n=n, valid_size=valid_size
        )
        actuals_train = preds_dict_train[KEY_ACTUAL]
        actuals_validation = preds_dict_valid[KEY_ACTUAL]

        synthetic_summary_train = pd.DataFrame(
            {
                label: [preds.mean(), mse(preds, actuals_train)]
                for label, preds in preds_dict_train.items()
                if KEY_GENERATED_DATA not in label.lower()
            },
            index=["ATE", "MSE"],
        ).T
        synthetic_summary_train["Abs % Error of ATE"] = np.abs(
            (
                synthetic_summary_train["ATE"]
                / synthetic_summary_train.loc[KEY_ACTUAL, "ATE"]
            )
            - 1
        )

        synthetic_summary_validation = pd.DataFrame(
            {
                label: [preds.mean(), mse(preds, actuals_validation)]
                for label, preds in preds_dict_valid.items()
                if KEY_GENERATED_DATA not in label.lower()
            },
            index=["ATE", "MSE"],
        ).T
        synthetic_summary_validation["Abs % Error of ATE"] = np.abs(
            (
                synthetic_summary_validation["ATE"]
                / synthetic_summary_validation.loc[KEY_ACTUAL, "ATE"]
            )
            - 1
        )

        # calculate kl divergence for training
        for label in synthetic_summary_train.index:
            stacked_values = np.hstack((preds_dict_train[label], actuals_train))
            stacked_low = np.percentile(stacked_values, 0.1)
            stacked_high = np.percentile(stacked_values, 99.9)
            bins = np.linspace(stacked_low, stacked_high, 100)

            distr = np.histogram(preds_dict_train[label], bins=bins)[0]
            distr = np.clip(distr / distr.sum(), 0.001, 0.999)
            true_distr = np.histogram(actuals_train, bins=bins)[0]
            true_distr = np.clip(true_distr / true_distr.sum(), 0.001, 0.999)

            kl = entropy(distr, true_distr)
            synthetic_summary_train.loc[label, "KL Divergence"] = kl

        # calculate kl divergence for validation
        for label in synthetic_summary_validation.index:
            stacked_values = np.hstack((preds_dict_valid[label], actuals_validation))
            stacked_low = np.percentile(stacked_values, 0.1)
            stacked_high = np.percentile(stacked_values, 99.9)
            bins = np.linspace(stacked_low, stacked_high, 100)

            distr = np.histogram(preds_dict_valid[label], bins=bins)[0]
            distr = np.clip(distr / distr.sum(), 0.001, 0.999)
            true_distr = np.histogram(actuals_validation, bins=bins)[0]
            true_distr = np.clip(true_distr / true_distr.sum(), 0.001, 0.999)

            kl = entropy(distr, true_distr)
            synthetic_summary_validation.loc[label, "KL Divergence"] = kl

        summaries_train.append(synthetic_summary_train)
        summaries_validation.append(synthetic_summary_validation)

    summary_train = sum(summaries_train) / k
    summary_validation = sum(summaries_validation) / k
    return (
        summary_train[["Abs % Error of ATE", "MSE", "KL Divergence"]],
        summary_validation[["Abs % Error of ATE", "MSE", "KL Divergence"]],
    )


def scatter_plot_summary_holdout(
    train_summary,
    validation_summary,
    k,
    label=["Train", "Validation"],
    drop_learners=[],
    drop_cols=[],
):
    """Generates a scatter plot comparing learner performance by training and validation.

    Args:
        train_summary (pd.DataFrame): summary for training synthetic data generated by get_synthetic_summary_holdout()
        validation_summary (pd.DataFrame): summary for validation synthetic data generated by
            get_synthetic_summary_holdout()
        label (string, optional): legend label for plot
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
    """
    train_summary = train_summary.drop(drop_learners).drop(drop_cols, axis=1)
    validation_summary = validation_summary.drop(drop_learners).drop(drop_cols, axis=1)

    plot_data = pd.concat([train_summary, validation_summary])
    plot_data["label"] = [i.replace("Train", "") for i in plot_data.index]
    plot_data["label"] = [i.replace("Validation", "") for i in plot_data.label]

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    xs = plot_data["Abs % Error of ATE"]
    ys = plot_data["MSE"]
    group = np.array(
        [label[0]] * train_summary.shape[0] + [label[1]] * validation_summary.shape[0]
    )
    cdict = {label[0]: "red", label[1]: "blue"}

    for g in np.unique(group):
        ix = np.where(group == g)[0].tolist()
        ax.scatter(xs[ix], ys[ix], c=cdict[g], label=g, s=100)

    for i, txt in enumerate(plot_data.label[:10]):
        ax.annotate(txt, (xs[i] + 0.005, ys[i]))

    ax.set_xlabel("Abs % Error of ATE")
    ax.set_ylabel("MSE")
    ax.set_title("Learner Performance (averaged over k={} simulations)".format(k))
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
    plt.show()


def bar_plot_summary_holdout(
    train_summary, validation_summary, k, drop_learners=[], drop_cols=[]
):
    """Generates a bar plot comparing learner performance by training and validation

    Args:
        train_summary (pd.DataFrame): summary for training synthetic data generated by get_synthetic_summary_holdout()
        validation_summary (pd.DataFrame): summary for validation synthetic data generated by
            get_synthetic_summary_holdout()
        k (int): number of simulations (used only for plot title text)
        drop_learners (list, optional): list of learners (str) to omit when plotting
        drop_cols (list, optional): list of metrics (str) to omit when plotting
    """
    train_summary = train_summary.drop([KEY_ACTUAL])
    train_summary["Learner"] = train_summary.index

    validation_summary = validation_summary.drop([KEY_ACTUAL])
    validation_summary["Learner"] = validation_summary.index

    for metric in ["Abs % Error of ATE", "MSE", "KL Divergence"]:
        plot_data_sub = pd.DataFrame(train_summary.Learner).reset_index(drop=True)
        plot_data_sub["train"] = train_summary[metric].values
        plot_data_sub["validation"] = validation_summary[metric].values
        plot_data_sub = plot_data_sub.set_index("Learner")
        plot_data_sub = plot_data_sub.drop(drop_learners).drop(drop_cols, axis=1)
        plot_data_sub = plot_data_sub.sort_values("train", ascending=True)

        plot_data_sub.plot(kind="bar", color=["red", "blue"], figsize=(12, 8))
        plt.xticks(rotation=30)
        plt.title(
            "Learner Performance of {} (averaged over k={} simulations)".format(
                metric, k
            )
        )


def get_synthetic_auuc(
    synthetic_preds,
    drop_learners=[],
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    plot=True,
):
    """Get auuc values for cumulative gains of model estimates in quantiles.

    For details, reference get_cumgain() and plot_gain()
    Args:
        synthetic_preds (dict): dictionary of predictions generated by get_synthetic_preds()
        or get_synthetic_preds_holdout()
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        plot (boolean,optional): plot the cumulative gain chart or not

    Returns:
        (pandas.DataFrame): auuc values by learner for cumulative gains of model estimates
    """
    synthetic_preds_df = synthetic_preds.copy()
    generated_data = synthetic_preds_df.pop(KEY_GENERATED_DATA)
    synthetic_preds_df = pd.DataFrame(synthetic_preds_df)
    synthetic_preds_df = synthetic_preds_df.drop(drop_learners, axis=1)

    synthetic_preds_df["y"] = generated_data[outcome_col]
    synthetic_preds_df["w"] = generated_data[treatment_col]
    if treatment_effect_col in generated_data.keys():
        synthetic_preds_df["tau"] = generated_data[treatment_effect_col]

    assert (
        (outcome_col in synthetic_preds_df.columns)
        and (treatment_col in synthetic_preds_df.columns)
        or treatment_effect_col in synthetic_preds_df.columns
    )

    cumlift = get_cumgain(
        synthetic_preds_df,
        outcome_col="y",
        treatment_col="w",
        treatment_effect_col="tau",
    )
    auuc_df = pd.DataFrame(cumlift.columns)
    auuc_df.columns = ["Learner"]
    auuc_df["cum_gain_auuc"] = [
        auc(cumlift.index.values / 100, cumlift[learner].values)
        for learner in cumlift.columns
    ]
    auuc_df = auuc_df.sort_values("cum_gain_auuc", ascending=False)

    if plot:
        plot_gain(
            synthetic_preds_df,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            treatment_effect_col=treatment_effect_col,
        )

    return auuc_df
