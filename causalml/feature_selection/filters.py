"""
Filter feature selection methods for uplift modeling

- Currently only for classification problem: the outcome variable of uplift model is binary.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer


class FilterSelect:
    """A class for feature importance methods."""

    def __init__(self):
        return

    @staticmethod
    def _filter_F_one_feature(data, treatment_indicator, feature_name, y_name, order=1):
        """
        Conduct F-test of the interaction between treatment and one feature.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            treatment_indicator (string): the column name for binary indicator of treatment (1) or control (0)
            feature_name (string): feature name, as one column in the data DataFrame
            y_name (string): name of the outcome variable
            order (int): the order of feature to be evaluated with the treatment effect, order takes 3 values: 1,2,3.
                order = 1 corresponds to linear importance of the feature, order=2 corresponds to quadratic and linear
                importance of the feature,
            order= 3 will calculate feature importance up to cubic forms.

        Returns:
            F_test_result : pd.DataFrame
                a data frame containing the feature importance statistics
        """
        Y = data[y_name]
        X = data[[treatment_indicator, feature_name]]
        X = sm.add_constant(X)
        X["{}-{}".format(treatment_indicator, feature_name)] = X[
            [treatment_indicator, feature_name]
        ].product(axis=1)

        if order not in [1, 2, 3]:
            raise Exception("ValueError: order argument only takes value 1,2,3.")

        if order == 1:
            pass
        elif order == 2:
            x_tmp_name = "{}_o{}".format(feature_name, order)
            X[x_tmp_name] = X[[feature_name]] ** order
            X["{}-{}".format(treatment_indicator, x_tmp_name)] = X[
                [treatment_indicator, x_tmp_name]
            ].product(axis=1)
        elif order == 3:
            x_tmp_name = "{}_o{}".format(feature_name, 2)
            X[x_tmp_name] = X[[feature_name]] ** 2
            X["{}-{}".format(treatment_indicator, x_tmp_name)] = X[
                [treatment_indicator, x_tmp_name]
            ].product(axis=1)

            x_tmp_name = "{}_o{}".format(feature_name, order)
            X[x_tmp_name] = X[[feature_name]] ** order
            X["{}-{}".format(treatment_indicator, x_tmp_name)] = X[
                [treatment_indicator, x_tmp_name]
            ].product(axis=1)

        model = sm.OLS(Y, X)
        result = model.fit()

        if order == 1:
            F_test = result.f_test(np.array([0, 0, 0, 1]))
        elif order == 2:
            F_test = result.f_test(np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]]))
        elif order == 3:
            F_test = result.f_test(
                np.array(
                    [
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                )
            )

        F_test_result = pd.DataFrame(
            {
                "feature": feature_name,  # for the interaction, not the main effect
                "method": "F{} Filter".format(order),
                "score": float(F_test.fvalue),
                "p_value": F_test.pvalue,
                "misc": "df_num: {}, df_denom: {}, order:{}".format(
                    F_test.df_num, F_test.df_denom, order
                ),
            },
            index=[0],
        ).reset_index(drop=True)

        return F_test_result

    def filter_F(self, data, treatment_indicator, features, y_name, order=1):
        """
        Rank features based on the F-statistics of the interaction.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            treatment_indicator (string): the column name for binary indicator of treatment (1) or control (0)
            features (list of string): list of feature names, that are columns in the data DataFrame
            y_name (string): name of the outcome variable
            order (int): the order of feature to be evaluated with the treatment effect, order takes 3 values: 1,2,3.
                order = 1 corresponds to linear importance of the feature, order=2 corresponds to quadratic and linear
                importance of the feature,
            order= 3 will calculate feature importance up to cubic forms.

        Returns:
            all_result : pd.DataFrame
                a data frame containing the feature importance statistics
        """
        if order not in [1, 2, 3]:
            raise Exception("ValueError: order argument only takes value 1,2,3.")

        all_result = pd.DataFrame()
        for x_name_i in features:
            one_result = self._filter_F_one_feature(
                data=data,
                treatment_indicator=treatment_indicator,
                feature_name=x_name_i,
                y_name=y_name,
                order=order,
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by="score", ascending=False)
        all_result["rank"] = all_result["score"].rank(ascending=False)

        return all_result

    @staticmethod
    def _filter_LR_one_feature(
        data, treatment_indicator, feature_name, y_name, order=1, disp=True
    ):
        """
        Conduct LR (Likelihood Ratio) test of the interaction between treatment and one feature.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            treatment_indicator (string): the column name for binary indicator of treatment (1) or control (0)
            feature_name (string): feature name, as one column in the data DataFrame
            y_name (string): name of the outcome variable
            order (int): the order of feature to be evaluated with the treatment effect, order takes 3 values: 1,2,3.
                order = 1 corresponds to linear importance of the feature, order=2 corresponds to quadratic and linear
                importance of the feature,
            order= 3 will calculate feature importance up to cubic forms.

        Returns:
            LR_test_result : pd.DataFrame
                a data frame containing the feature importance statistics
        """
        Y = data[y_name]

        # Restricted model
        x_name_r = ["const", treatment_indicator, feature_name]
        x_name_f = x_name_r.copy()
        X = data[[treatment_indicator, feature_name]]
        X = sm.add_constant(X)

        X["{}-{}".format(treatment_indicator, feature_name)] = X[
            [treatment_indicator, feature_name]
        ].product(axis=1)
        x_name_f.append("{}-{}".format(treatment_indicator, feature_name))

        if order == 2:
            x_tmp_name = "{}_o{}".format(feature_name, order)
            X[x_tmp_name] = X[[feature_name]] ** order
            X["{}-{}".format(treatment_indicator, x_tmp_name)] = X[
                [treatment_indicator, x_tmp_name]
            ].product(axis=1)
            x_name_r.append(x_tmp_name)
            x_name_f += [x_tmp_name, "{}-{}".format(treatment_indicator, x_tmp_name)]
        elif order == 3:
            x_tmp_name = "{}_o{}".format(feature_name, 2)
            X[x_tmp_name] = X[[feature_name]] ** 2
            X["{}-{}".format(treatment_indicator, x_tmp_name)] = X[
                [treatment_indicator, x_tmp_name]
            ].product(axis=1)
            x_name_r.append(x_tmp_name)
            x_name_f += [x_tmp_name, "{}-{}".format(treatment_indicator, x_tmp_name)]
            x_tmp_name = "{}_o{}".format(feature_name, order)
            X[x_tmp_name] = X[[feature_name]] ** order
            X["{}-{}".format(treatment_indicator, x_tmp_name)] = X[
                [treatment_indicator, x_tmp_name]
            ].product(axis=1)
            x_name_r.append(x_tmp_name)
            x_name_f += [x_tmp_name, "{}-{}".format(treatment_indicator, x_tmp_name)]

        # Full model (with interaction)
        model_r = sm.Logit(Y, X[x_name_r])
        result_r = model_r.fit(disp=disp)

        model_f = sm.Logit(Y, X[x_name_f])
        result_f = model_f.fit(disp=disp)

        LR_stat = -2 * (result_r.llf - result_f.llf)
        LR_df = len(result_f.params) - len(result_r.params)
        LR_pvalue = 1 - stats.chi2.cdf(LR_stat, df=LR_df)

        LR_test_result = pd.DataFrame(
            {
                "feature": feature_name,  # for the interaction, not the main effect
                "method": "LR{} Filter".format(order),
                "score": LR_stat,
                "p_value": LR_pvalue,
                "misc": "df: {}, order: {}".format(LR_df, order),
            },
            index=[0],
        ).reset_index(drop=True)

        return LR_test_result

    def filter_LR(
        self, data, treatment_indicator, features, y_name, order=1, disp=True
    ):
        """
        Rank features based on the LRT-statistics of the interaction.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            treatment_indicator (string): the column name for binary indicator of treatment (1) or control (0)
            feature_name (string): feature name, as one column in the data DataFrame
            y_name (string): name of the outcome variable
            order (int): the order of feature to be evaluated with the treatment effect, order takes 3 values: 1,2,3.
                order = 1 corresponds to linear importance of the feature, order=2 corresponds to quadratic and linear
                importance of the feature,
            order= 3 will calculate feature importance up to cubic forms.

        Returns:
            all_result : pd.DataFrame
                a data frame containing the feature importance statistics
        """
        if order not in [1, 2, 3]:
            raise Exception("ValueError: order argument only takes value 1,2,3.")

        all_result = pd.DataFrame()
        for x_name_i in features:
            one_result = self._filter_LR_one_feature(
                data=data,
                treatment_indicator=treatment_indicator,
                feature_name=x_name_i,
                y_name=y_name,
                order=order,
                disp=disp,
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by="score", ascending=False)
        all_result["rank"] = all_result["score"].rank(ascending=False)

        return all_result

    # Get node summary - a function
    @staticmethod
    def _GetNodeSummary(
        data,
        experiment_group_column="treatment_group_key",
        y_name="conversion",
        smooth=True,
    ):
        """
        To count the conversions and get the probabilities by treatment groups. This function comes from the uplift
        tree algorithm, that is used for tree node split evaluation.

        Parameters
        ----------
        data : DataFrame
            The DataFrame that contains all the data (in the current "node").
        experiment_group_column : str
            Treatment indicator column name.
        y_name : str
            Label indicator column name.
        smooth : bool
            Smooth label count by adding 1 in case certain labels do not occur
            naturally with a treatment. Prevents zero divisions.

        Returns
        -------
        results : dict
            Counts of conversions by treatment groups, of the form:
            {'control': {0: 10, 1: 8}, 'treatment1': {0: 5, 1: 15}}
        nodeSummary: dict
            Probability of conversion and group size by treatment groups, of
            the form:
            {'control': [0.490, 500], 'treatment1': [0.584, 500]}
        """

        # Note: results and nodeSummary are both dict with treatment_group_key
        # as the key.  So we can compute the treatment effect and/or
        # divergence easily.

        # Counts of conversions by treatment group
        results_series = data.groupby([experiment_group_column, y_name]).size()

        treatment_group_keys = results_series.index.levels[0].tolist()
        y_name_keys = results_series.index.levels[1].tolist()

        results = {}
        for ti in treatment_group_keys:
            results.update({ti: {}})
            for ci in y_name_keys:
                if smooth:
                    results[ti].update(
                        {
                            ci: (
                                results_series[ti, ci]
                                if results_series.index.isin([(ti, ci)]).any()
                                else 1
                            )
                        }
                    )
                else:
                    results[ti].update({ci: results_series[ti, ci]})

        # Probability of conversion and group size by treatment group
        nodeSummary = {}
        for treatment_group_key in results:
            n_1 = results[treatment_group_key].get(1, 0)
            n_total = results[treatment_group_key].get(1, 0) + results[
                treatment_group_key
            ].get(0, 0)
            y_mean = 1.0 * n_1 / n_total
            nodeSummary[treatment_group_key] = [y_mean, n_total]

        return results, nodeSummary

    # Divergence-related functions, from upliftpy
    @staticmethod
    def _kl_divergence(pk, qk):
        """
        Calculate KL Divergence for binary classification.

        Args:
            pk (float): Probability of class 1 in treatment group
            qk (float): Probability of class 1 in control group
        """
        if qk < 0.1**6:
            qk = 0.1**6
        elif qk > 1 - 0.1**6:
            qk = 1 - 0.1**6
        S = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))
        return S

    def _evaluate_KL(self, nodeSummary, control_group="control"):
        """
        Calculate the multi-treatment unconditional D (one node)
        with KL Divergence as split Evaluation function.

        Args:
            nodeSummary (dict): a dictionary containing the statistics for a tree node sample
            control_group (string, optional, default='control'): the name for control group

        Notes
        -----
        The function works for more than one non-control treatment groups.
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += self._kl_divergence(nodeSummary[treatment_group][0], pc)
        return d_res

    @staticmethod
    def _evaluate_ED(nodeSummary, control_group="control"):
        """
        Calculate the multi-treatment unconditional D (one node)
        with Euclidean Distance as split Evaluation function.

        Args:
            nodeSummary (dict): a dictionary containing the statistics for a tree node sample
            control_group (string, optional, default='control'): the name for control group
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += 2 * (nodeSummary[treatment_group][0] - pc) ** 2
        return d_res

    @staticmethod
    def _evaluate_Chi(nodeSummary, control_group="control"):
        """
        Calculate the multi-treatment unconditional D (one node)
        with Chi-Square as split Evaluation function.

        Args:
            nodeSummary (dict): a dictionary containing the statistics for a tree node sample
            control_group (string, optional, default='control'): the name for control group
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += (nodeSummary[treatment_group][0] - pc) ** 2 / max(
                    0.1**6, pc
                ) + (nodeSummary[treatment_group][0] - pc) ** 2 / max(0.1**6, 1 - pc)
        return d_res

    def _filter_D_one_feature(
        self,
        data,
        feature_name,
        y_name,
        n_bins=10,
        method="KL",
        control_group="control",
        experiment_group_column="treatment_group_key",
        null_impute=None,
    ):
        """
        Calculate the chosen divergence measure for one feature.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            treatment_indicator (string): the column name for binary indicator of treatment (1) or control (0)
            feature_name (string): feature name, as one column in the data DataFrame
            y_name (string): name of the outcome variable
            method (string, optional, default = 'KL'): taking one of the following values {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.
                'F' for F-test
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods, KL divergence, Euclidean distance,
                Chi-Square respectively
            experiment_group_column (string, optional, default = 'treatment_group_key'): the experiment column name in
                the DataFrame, which contains the treatment and control assignment label
            control_group (string, optional, default = 'control'): name for control group, value in the experiment
                group column
            n_bins (int, optional, default = 10): number of bins to be used for bin-based uplift filter methods
            null_impute (str, optional, default=None): impute np.nan present in the data taking on of the following
                strategy values {'mean', 'median', 'most_frequent', None}. If Value is None and null is present then
                exception will be raised

        Returns:
            D_result : pd.DataFrame
                a data frame containing the feature importance statistics
        """
        # [TODO] Application to categorical features

        if method == "KL":
            evaluationFunction = self._evaluate_KL
        elif method == "ED":
            evaluationFunction = self._evaluate_ED
        elif method == "Chi":
            evaluationFunction = self._evaluate_Chi

        totalSize = len(data.index)

        # impute null if enabled
        if null_impute is not None:
            data[feature_name] = SimpleImputer(
                missing_values=np.nan, strategy=null_impute
            ).fit_transform(data[feature_name].values.reshape(-1, 1))
        elif data[feature_name].isna().any():
            raise Exception(
                "Null value(s) present in column '{}'. Please impute the null value or use null_impute parameter "
                "provided.".format(feature_name)
            )

        # drop duplicate edges in pq.cut result to avoid issues
        x_bin = pd.qcut(
            data[feature_name].values, n_bins, labels=False, duplicates="drop"
        )

        d_children = 0

        for i_bin in range(np.nanmax(x_bin).astype(int) + 1):  # range(n_bins):
            nodeSummary = self._GetNodeSummary(
                data=data.loc[x_bin == i_bin],
                experiment_group_column=experiment_group_column,
                y_name=y_name,
            )[1]
            nodeScore = evaluationFunction(nodeSummary, control_group=control_group)
            nodeSize = sum([x[1] for x in list(nodeSummary.values())])
            d_children += nodeScore * nodeSize / totalSize

        parentNodeSummary = self._GetNodeSummary(
            data=data, experiment_group_column=experiment_group_column, y_name=y_name
        )[1]
        d_parent = evaluationFunction(parentNodeSummary, control_group=control_group)

        d_res = d_children - d_parent

        D_result = pd.DataFrame(
            {
                "feature": feature_name,
                "method": method,
                "score": d_res,
                "p_value": None,
                "misc": "number_of_bins: {}".format(
                    min(n_bins, np.nanmax(x_bin).astype(int) + 1)
                ),  # format(n_bins),
            },
            index=[0],
        ).reset_index(drop=True)

        return D_result

    def filter_D(
        self,
        data,
        features,
        y_name,
        n_bins=10,
        method="KL",
        control_group="control",
        experiment_group_column="treatment_group_key",
        null_impute=None,
    ):
        """
        Rank features based on the chosen divergence measure.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            treatment_indicator (string): the column name for binary indicator of treatment (1) or control (0)
            features (list of string): list of feature names, that are columns in the data DataFrame
            y_name (string): name of the outcome variable
            method (string, optional, default = 'KL'): taking one of the following values {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.
                'F' for F-test
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods, KL divergence, Euclidean distance, Chi-Square
                respectively
            experiment_group_column (string, optional, default = 'treatment_group_key'): the experiment column name in
                the DataFrame, which contains the treatment and control assignment label
            control_group (string, optional, default = 'control'): name for control group, value in the experiment
                group column
            n_bins (int, optional, default = 10): number of bins to be used for bin-based uplift filter methods
            null_impute (str, optional, default=None): impute np.nan present in the data taking on of the followin
                strategy values {'mean', 'median', 'most_frequent', None}. If Value is None and null is present then
                exception will be raised

        Returns:
            all_result : pd.DataFrame
                a data frame containing the feature importance statistics
        """

        all_result = pd.DataFrame()

        for x_name_i in features:
            one_result = self._filter_D_one_feature(
                data=data,
                feature_name=x_name_i,
                y_name=y_name,
                n_bins=n_bins,
                method=method,
                control_group=control_group,
                experiment_group_column=experiment_group_column,
                null_impute=null_impute,
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by="score", ascending=False)
        all_result["rank"] = all_result["score"].rank(ascending=False)

        return all_result

    def get_importance(
        self,
        data,
        features,
        y_name,
        method,
        experiment_group_column="treatment_group_key",
        control_group="control",
        treatment_group="treatment",
        n_bins=5,
        null_impute=None,
        order=1,
        disp=False,
    ):
        """
        Rank features based on the chosen statistic of the interaction.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            features (list of string): list of feature names, that are columns in the data DataFrame
            y_name (string): name of the outcome variable
            method (string, optional, default = 'KL'): taking one of the following values {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.
                'F' for F-test
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods, KL divergence, Euclidean distance, Chi-Square
                respectively
            experiment_group_column (string): the experiment column name in the DataFrame, which contains the treatment
                and control assignment label
            control_group (string): name for control group, value in the experiment group column
            treatment_group (string): name for treatment group, value in the experiment group column
            n_bins (int, optional): number of bins to be used for bin-based uplift filter methods
            null_impute (str, optional, default=None): impute np.nan present in the data taking on of the following
                strategy values {'mean', 'median', 'most_frequent', None}. If value is None and null is present then
                exception will be raised
            order (int): the order of feature to be evaluated with the treatment effect for F filter and LR filter,
                order takes 3 values: 1,2,3. order = 1 corresponds to linear importance of the feature, order=2
                corresponds to quadratic and linear importance of the feature,
            order= 3 will calculate feature importance up to cubic forms.
            disp (bool): Set to True to print convergence messages for Logistic regression convergence in LR method.

        Returns:
            all_result : pd.DataFrame
                a data frame with following columns: ['method', 'feature', 'rank', 'score', 'p_value', 'misc']
        """

        if method == "F":
            data = data[
                data[experiment_group_column].isin([control_group, treatment_group])
            ]
            data["treatment_indicator"] = 0
            data.loc[
                data[experiment_group_column] == treatment_group, "treatment_indicator"
            ] = 1
            all_result = self.filter_F(
                data=data,
                treatment_indicator="treatment_indicator",
                features=features,
                y_name=y_name,
                order=order,
            )
        elif method == "LR":
            data = data[
                data[experiment_group_column].isin([control_group, treatment_group])
            ]
            data["treatment_indicator"] = 0
            data.loc[
                data[experiment_group_column] == treatment_group, "treatment_indicator"
            ] = 1
            all_result = self.filter_LR(
                data=data,
                disp=disp,
                treatment_indicator="treatment_indicator",
                features=features,
                y_name=y_name,
                order=order,
            )
        else:
            all_result = self.filter_D(
                data=data,
                method=method,
                features=features,
                y_name=y_name,
                n_bins=n_bins,
                control_group=control_group,
                experiment_group_column=experiment_group_column,
                null_impute=null_impute,
            )

        all_result["method"] = method + " filter"
        return all_result[["method", "feature", "rank", "score", "p_value", "misc"]]
