import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module

logger = logging.getLogger("sensitivity")

SUMMARY_COLS = ["Method", "ATE", "New ATE", "New ATE LB", "New ATE UB"]


def one_sided(alpha, p, treatment):
    """One sided confounding function.
    Reference:  Blackwell, Matthew. "A selection bias approach to sensitivity analysis
    for causal effects." Political Analysis 22.2 (2014): 169-182.
    https://www.mattblackwell.org/files/papers/causalsens.pdf

    Args:
        alpha (np.array): a confounding values vector
        p (np.array): a propensity score vector between 0 and 1
        treatment (np.array): a treatment vector (1 if treated, otherwise 0)
    """
    assert p.shape[0] == treatment.shape[0]
    adj = alpha * (1 - p) * treatment - alpha * p * (1 - treatment)
    return adj


def alignment(alpha, p, treatment):
    """Alignment confounding function.
    Reference:  Blackwell, Matthew. "A selection bias approach to sensitivity analysis
    for causal effects." Political Analysis 22.2 (2014): 169-182.
    https://www.mattblackwell.org/files/papers/causalsens.pdf

    Args:
        alpha (np.array): a confounding values vector
        p (np.array): a propensity score vector between 0 and 1
        treatment (np.array): a treatment vector (1 if treated, otherwise 0)
    """

    assert p.shape[0] == treatment.shape[0]
    adj = alpha * (1 - p) * treatment + alpha * p * (1 - treatment)
    return adj


def one_sided_att(alpha, p, treatment):
    """One sided confounding function for the average effect of the treatment among the treated units (ATT)

    Reference:  Blackwell, Matthew. "A selection bias approach to sensitivity analysis
    for causal effects." Political Analysis 22.2 (2014): 169-182.
    https://www.mattblackwell.org/files/papers/causalsens.pdf

    Args:
        alpha (np.array): a confounding values vector
        p (np.array): a propensity score vector between 0 and 1
        treatment (np.array): a treatment vector (1 if treated, otherwise 0)
    """
    assert p.shape[0] == treatment.shape[0]
    adj = alpha * (1 - treatment)
    return adj


def alignment_att(alpha, p, treatment):
    """Alignment confounding function for the average effect of the treatment among the treated units (ATT)

    Reference:  Blackwell, Matthew. "A selection bias approach to sensitivity analysis
    for causal effects." Political Analysis 22.2 (2014): 169-182.
    https://www.mattblackwell.org/files/papers/causalsens.pdf

    Args:
        alpha (np.array): a confounding values vector
        p (np.array): a propensity score vector between 0 and 1
        treatment (np.array): a treatment vector (1 if treated, otherwise 0)
    """
    assert p.shape[0] == treatment.shape[0]
    adj = alpha * (1 - treatment)
    return adj


class Sensitivity:
    """A Sensitivity Check class to support Placebo Treatment, Irrelevant Additional Confounder
    and Subset validation refutation methods to verify causal inference.

    Reference: https://github.com/microsoft/dowhy/blob/master/dowhy/causal_refuters/
    """

    def __init__(
        self,
        df,
        inference_features,
        p_col,
        treatment_col,
        outcome_col,
        learner,
        *args,
        **kwargs,
    ):
        """Initialize.

        Args:
            df (pd.DataFrame): input data frame
            inferenece_features (list of str): a list of columns that used in learner for inference
            p_col (str): column name of propensity score
            treatment_col (str): column name of whether in treatment of control
            outcome_col (str): column name of outcome
            learner (model): a model to estimate outcomes and treatment effects
        """

        self.df = df
        self.inference_features = inference_features
        self.p_col = p_col
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.learner = learner

    def get_prediction(self, X, p, treatment, y):
        """Return the treatment effects prediction.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity score vector between 0 and 1
            treatment (np.array): a treatment vector (1 if treated, otherwise 0)
            y (np.array): an outcome vector
        Returns:
            (numpy.ndarray): Predictions of treatment effects
        """

        learner = self.learner
        try:
            preds = learner.fit_predict(X=X, p=p, treatment=treatment, y=y).flatten()
        except TypeError:
            preds = learner.fit_predict(X=X, treatment=treatment, y=y).flatten()
        return preds

    def get_ate_ci(self, X, p, treatment, y):
        """Return the confidence intervals for treatment effects prediction.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity score vector between 0 and 1
            treatment (np.array): a treatment vector (1 if treated, otherwise 0)
            y (np.array): an outcome vector
        Returns:
            (numpy.ndarray): Mean and confidence interval (LB, UB) of the ATE estimate.
        """

        try:
            ate, ate_lower, ate_upper = self.learner.estimate_ate(
                X=X, p=p, treatment=treatment, y=y, return_ci=True
            )
        except TypeError:
            ate, ate_lower, ate_upper = self.learner.estimate_ate(
                X=X, p=p, treatment=treatment, y=y
            )
        return ate[0], ate_lower[0], ate_upper[0]

    @staticmethod
    def get_class_object(method_name, *args, **kwargs):
        """Return class object based on input method
        Args:
            method_name (list of str): a list of sensitivity analysis method
        Returns:
            (class): Sensitivy Class
        """

        method_list = [
            "Placebo Treatment",
            "Random Cause",
            "Subset Data",
            "Random Replace",
            "Selection Bias",
        ]
        class_name = "Sensitivity" + method_name.replace(" ", "")

        try:
            getattr(import_module("causalml.metrics.sensitivity"), class_name)
            return getattr(import_module("causalml.metrics.sensitivity"), class_name)
        except AttributeError:
            raise AttributeError(
                "{} is not an existing method for sensitiviy analysis.".format(
                    method_name
                )
                + " Select one of {}".format(method_list)
            )

    def sensitivity_analysis(
        self, methods, sample_size=None, confound="one_sided", alpha_range=None
    ):
        """Return the sensitivity data by different method

        Args:
            method (list of str): a list of sensitivity analysis method
            sample_size (float, optional): ratio for subset the original data
            confound (string, optional): the name of confouding function
            alpha_range (np.array, optional): a parameter to pass the confounding function

        Returns:
            X (np.matrix): a feature matrix
            p (np.array): a propensity score vector between 0 and 1
            treatment (np.array): a treatment vector (1 if treated, otherwise 0)
            y (np.array): an outcome vector
        """
        if alpha_range is None:
            y = self.df[self.outcome_col]
            iqr = y.quantile(0.75) - y.quantile(0.25)
            alpha_range = np.linspace(-iqr / 2, iqr / 2, 11)
            if 0 not in alpha_range:
                alpha_range = np.append(alpha_range, 0)
        else:
            alpha_range = alpha_range

        alpha_range.sort()

        summary = []
        for method in methods:
            sens = self.get_class_object(method)
            sens = sens(
                self.df,
                self.inference_features,
                self.p_col,
                self.treatment_col,
                self.outcome_col,
                self.learner,
                sample_size=sample_size,
                confound=confound,
                alpha_range=alpha_range,
            )

            if method == "Subset Data":
                method = method + "(sample size @{})".format(sample_size)

            sens_df = sens.summary(method=method)
            summary.append(sens_df.values.tolist()[0])

        summary_df = pd.DataFrame(summary, columns=SUMMARY_COLS)

        return summary_df

    def summary(self, method):
        """Summary report
        Args:
            method_name (str): sensitivity analysis method

        Returns:
            (pd.DataFrame): a summary dataframe
        """
        method_name = method

        X = self.df[self.inference_features].values
        p = self.df[self.p_col].values
        treatment = self.df[self.treatment_col].values
        y = self.df[self.outcome_col].values

        preds = self.get_prediction(X, p, treatment, y)
        ate = preds.mean()
        ate_new, ate_new_lower, ate_new_upper = self.sensitivity_estimate()

        sensitivity_summary = pd.DataFrame(
            [method_name, ate, ate_new, ate_new_lower, ate_new_upper]
        ).T
        sensitivity_summary.columns = SUMMARY_COLS
        return sensitivity_summary

    def sensitivity_estimate(self):
        raise NotImplementedError


class SensitivityPlaceboTreatment(Sensitivity):
    """Replaces the treatment variable with a new variable randomly generated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sensitivity_estimate(self):
        """Summary report
        Args:
           return_ci (str): sensitivity analysis method

        Returns:
            (pd.DataFrame): a summary dataframe
        """
        num_rows = self.df.shape[0]

        X = self.df[self.inference_features].values
        p = self.df[self.p_col].values
        treatment_new = np.random.randint(2, size=num_rows)
        y = self.df[self.outcome_col].values

        ate_new, ate_new_lower, ate_new_upper = self.get_ate_ci(X, p, treatment_new, y)
        return ate_new, ate_new_lower, ate_new_upper


class SensitivityRandomCause(Sensitivity):
    """Adds an irrelevant random covariate to the dataframe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sensitivity_estimate(self):
        num_rows = self.df.shape[0]
        new_data = np.random.randn(num_rows)

        X = self.df[self.inference_features].values
        p = self.df[self.p_col].values
        treatment = self.df[self.treatment_col].values
        y = self.df[self.outcome_col].values
        X_new = np.hstack((X, new_data.reshape((-1, 1))))

        ate_new, ate_new_lower, ate_new_upper = self.get_ate_ci(X_new, p, treatment, y)
        return ate_new, ate_new_lower, ate_new_upper


class SensitivityRandomReplace(Sensitivity):
    """Replaces a random covariate with an irrelevant variable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "replaced_feature" not in kwargs:
            replaced_feature_index = np.random.randint(len(self.inference_features))
            self.replaced_feature = self.inference_features[replaced_feature_index]
        else:
            self.replaced_feature = kwargs["replaced_feature"]

    def sensitivity_estimate(self):
        """Replaces a random covariate with an irrelevant variable."""

        logger.info(
            "Replace feature {} with an random irrelevant variable".format(
                self.replaced_feature
            )
        )
        df_new = self.df.copy()
        num_rows = self.df.shape[0]
        df_new[self.replaced_feature] = np.random.randn(num_rows)

        X_new = df_new[self.inference_features].values
        p_new = df_new[self.p_col].values
        treatment_new = df_new[self.treatment_col].values
        y_new = df_new[self.outcome_col].values

        ate_new, ate_new_lower, ate_new_upper = self.get_ate_ci(
            X_new, p_new, treatment_new, y_new
        )
        return ate_new, ate_new_lower, ate_new_upper


class SensitivitySubsetData(Sensitivity):
    """Takes a random subset of size sample_size of the data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = kwargs["sample_size"]
        assert self.sample_size is not None

    def sensitivity_estimate(self):
        df_new = self.df.sample(frac=self.sample_size).copy()

        X_new = df_new[self.inference_features].values
        p_new = df_new[self.p_col].values
        treatment_new = df_new[self.treatment_col].values
        y_new = df_new[self.outcome_col].values

        ate_new, ate_new_lower, ate_new_upper = self.get_ate_ci(
            X_new, p_new, treatment_new, y_new
        )
        return ate_new, ate_new_lower, ate_new_upper


class SensitivitySelectionBias(Sensitivity):
    """Reference:

    [1] Blackwell, Matthew. "A selection bias approach to sensitivity analysis
    for causal effects." Political Analysis 22.2 (2014): 169-182.
    https://www.mattblackwell.org/files/papers/causalsens.pdf

    [2] Confouding parameter alpha_range using the same range as in:
    https://github.com/mattblackwell/causalsens/blob/master/R/causalsens.R

    """

    def __init__(
        self,
        *args,
        confound="one_sided",
        alpha_range=None,
        sensitivity_features=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        """Initialize.

        Args:
            confound (string): the name of confouding function
            alpha_range (np.array): a parameter to pass the confounding function
            sensitivity_features (list of str): ): a list of columns that to check each individual partial r-square
        """

        logger.info("Only works for linear outcome models right now. Check back soon.")
        confounding_functions = {
            "one_sided": one_sided,
            "alignment": alignment,
            "one_sided_att": one_sided_att,
            "alignment_att": alignment_att,
        }

        try:
            confound_func = confounding_functions[confound]
        except KeyError:
            raise NotImplementedError(
                f"Confounding function, {confound} is not implemented. \
                                        Use one of {confounding_functions.keys()}"
            )

        self.confound = confound_func

        if sensitivity_features is None:
            self.sensitivity_features = self.inference_features
        else:
            self.sensitivity_features = sensitivity_features

        if alpha_range is None:
            y = self.df[self.outcome_col]
            iqr = y.quantile(0.75) - y.quantile(0.25)
            self.alpha_range = np.linspace(-iqr / 2, iqr / 2, 11)
            if 0 not in self.alpha_range:
                self.alpha_range = np.append(self.alpha_range, 0)
        else:
            self.alpha_range = alpha_range

        self.alpha_range.sort()

    def causalsens(self):
        alpha_range = self.alpha_range
        confound = self.confound
        df = self.df
        X = df[self.inference_features].values
        p = df[self.p_col].values
        treatment = df[self.treatment_col].values
        y = df[self.outcome_col].values

        preds = self.get_prediction(X, p, treatment, y)

        sens_df = pd.DataFrame()

        sens = []
        for a in alpha_range:
            adj = confound(a, p, treatment)
            preds_adj = y - adj
            s_preds = self.get_prediction(X, p, treatment, preds_adj)
            ate, ate_lb, ate_ub = self.get_ate_ci(X, p, treatment, preds_adj)

            s_preds_residul = preds_adj - s_preds
            rsqs = a**2 * np.var(treatment) / np.var(s_preds_residul)

            sens.append([a, rsqs, ate, ate_lb, ate_ub])

        sens_df = pd.DataFrame(
            sens, columns=["alpha", "rsqs", "New ATE", "New ATE LB", "New ATE UB"]
        )

        rss = np.sum(np.square(y - preds))
        partial_rsqs = []
        for feature in self.sensitivity_features:
            df_new = df.copy()
            X_new = df_new[self.inference_features].drop(feature, axis=1).copy()
            y_new_preds = self.get_prediction(X_new, p, treatment, y)
            rss_new = np.sum(np.square(y - y_new_preds))
            partial_rsqs.append(((rss_new - rss) / rss))

        partial_rsqs_df = pd.DataFrame([self.sensitivity_features, partial_rsqs]).T
        partial_rsqs_df.columns = ["feature", "partial_rsqs"]

        return sens_df, partial_rsqs_df

    def summary(self, method="Selection Bias"):
        """Summary report for Selection Bias Method
        Args:
            method_name (str): sensitivity analysis method
        Returns:
            (pd.DataFrame): a summary dataframe
        """

        method_name = method
        sensitivity_summary = self.causalsens()[0]
        sensitivity_summary["Method"] = [
            method_name + " (alpha@" + str(round(i, 5)) + ", with r-sqaure:"
            for i in sensitivity_summary.alpha
        ]
        sensitivity_summary["Method"] = sensitivity_summary[
            "Method"
        ] + sensitivity_summary["rsqs"].round(5).astype(str)
        sensitivity_summary["ATE"] = sensitivity_summary[
            sensitivity_summary.alpha == 0
        ]["New ATE"]
        return sensitivity_summary[SUMMARY_COLS]

    @staticmethod
    def plot(sens_df, partial_rsqs_df=None, type="raw", ci=False, partial_rsqs=False):
        """Plot the results of a sensitivity analysis against unmeasured
        Args:
            sens_df (pandas.DataFrame): a data frame output from causalsens
            partial_rsqs_d (pandas.DataFrame) : a data frame output from causalsens including partial rsqure
            type (str, optional): the type of plot to draw, 'raw' or 'r.squared' are supported
            ci (bool, optional): whether plot confidence intervals
            partial_rsqs (bool, optional): whether plot partial rsquare results
        """

        if type == "raw" and not ci:
            fig, ax = plt.subplots()
            y_max = round(sens_df["New ATE UB"].max() * 1.1, 4)
            y_min = round(sens_df["New ATE LB"].min() * 0.9, 4)
            x_max = round(sens_df.alpha.max() * 1.1, 4)
            x_min = round(sens_df.alpha.min() * 0.9, 4)
            plt.ylim(y_min, y_max)
            plt.xlim(x_min, x_max)
            ax.plot(sens_df.alpha, sens_df["New ATE"])
        elif type == "raw" and ci:
            fig, ax = plt.subplots()
            y_max = round(sens_df["New ATE UB"].max() * 1.1, 4)
            y_min = round(sens_df["New ATE LB"].min() * 0.9, 4)
            x_max = round(sens_df.alpha.max() * 1.1, 4)
            x_min = round(sens_df.alpha.min() * 0.9, 4)
            plt.ylim(y_min, y_max)
            plt.xlim(x_min, x_max)
            ax.fill_between(
                sens_df.alpha,
                sens_df["New ATE LB"],
                sens_df["New ATE UB"],
                color="gray",
                alpha=0.5,
            )
            ax.plot(sens_df.alpha, sens_df["New ATE"])
        elif type == "r.squared" and ci:
            fig, ax = plt.subplots()
            y_max = round(sens_df["New ATE UB"].max() * 1.1, 4)
            y_min = round(sens_df["New ATE LB"].min() * 0.9, 4)
            plt.ylim(y_min, y_max)
            ax.fill_between(
                sens_df.rsqs,
                sens_df["New ATE LB"],
                sens_df["New ATE UB"],
                color="gray",
                alpha=0.5,
            )
            ax.plot(sens_df.rsqs, sens_df["New ATE"])
            if partial_rsqs:
                plt.scatter(
                    partial_rsqs_df.partial_rsqs,
                    list(sens_df[sens_df.alpha == 0]["New ATE"])
                    * partial_rsqs_df.shape[0],
                    marker="x",
                    color="red",
                    linewidth=10,
                )
        elif type == "r.squared" and not ci:
            fig, ax = plt.subplots()
            y_max = round(sens_df["New ATE UB"].max() * 1.1, 4)
            y_min = round(sens_df["New ATE LB"].min() * 0.9, 4)
            plt.ylim(y_min, y_max)
            plt.plot(sens_df.rsqs, sens_df["New ATE"])
            if partial_rsqs:
                plt.scatter(
                    partial_rsqs_df.partial_rsqs,
                    list(sens_df[sens_df.alpha == 0]["New ATE"])
                    * partial_rsqs_df.shape[0],
                    marker="x",
                    color="red",
                    linewidth=10,
                )

    @staticmethod
    def partial_rsqs_confounding(sens_df, feature_name, partial_rsqs_value, range=0.01):
        """Check partial rsqs values of feature corresponding confounding amonunt of ATE
        Args:
            sens_df (pandas.DataFrame): a data frame output from causalsens
            feature_name (str): feature name to check
            partial_rsqs_value (float) : partial rsquare value of feature
            range (float) : range to search from sens_df

        Return: min and max value of confounding amount
        """

        rsqs_dict = []
        for i in sens_df.rsqs:
            if (
                partial_rsqs_value - partial_rsqs_value * range
                < i
                < partial_rsqs_value + partial_rsqs_value * range
            ):
                rsqs_dict.append(i)

        if rsqs_dict:
            confounding_min = sens_df[sens_df.rsqs.isin(rsqs_dict)].alpha.min()
            confounding_max = sens_df[sens_df.rsqs.isin(rsqs_dict)].alpha.max()
            logger.info(
                "Only works for linear outcome models right now. Check back soon."
            )
            logger.info(
                "For feature {} with partial rsquare {} confounding amount with possible values: {}, {}".format(
                    feature_name, partial_rsqs_value, confounding_min, confounding_max
                )
            )
            return [confounding_min, confounding_max]
        else:
            logger.info(
                "Cannot find correponding rsquare value within the range for input, please edit confounding",
                "values vector or use a larger range and try again",
            )
