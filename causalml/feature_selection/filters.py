"""
Filter feature selection methods for uplift modeling

- Currently only for classification problem: the outcome variable of uplift model is binary.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
    
class FilterSelect:
    """A class for feature importance methods.
    """

    def __init__(self):
        return

    @staticmethod
    def _filter_F_one_feature(data, treatment_indicator, feature_name, y_name):
        """
        Conduct F-test of the interaction between treatment and one feature.

        Parameters
        ----------
        data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
        treatment_indicator (string): the column name for binary indicator of treatment (value 1) or control (value 0) 
        feature_name (string): feature name, as one column in the data DataFrame
        y_name (string): name of the outcome variable

        Returns
        ----------
        (pd.DataFrame): a data frame containing the feature importance statistics
        """
        Y = data[y_name]
        X = data[[treatment_indicator, feature_name]]
        X = sm.add_constant(X)
        X['{}-{}'.format(treatment_indicator, feature_name)] = X[[treatment_indicator, feature_name]].product(axis=1)

        model = sm.OLS(Y, X)
        result = model.fit()

        F_test = result.f_test(np.array([0, 0, 0, 1]))
        F_test_result = pd.DataFrame({
            'feature': feature_name, # for the interaction, not the main effect
            'method': 'F-statistic',
            'score': F_test.fvalue[0][0], 
            'p_value': F_test.pvalue, 
            'misc': 'df_num: {}, df_denom: {}'.format(F_test.df_num, F_test.df_denom), 
        }, index=[0]).reset_index(drop=True)

        return F_test_result


    def filter_F(self, data, treatment_indicator, features, y_name):
        """
        Rank features based on the F-statistics of the interaction.

        Parameters
        ----------
        data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
        treatment_indicator (string): the column name for binary indicator of treatment (value 1) or control (value 0) 
        features (list of string): list of feature names, that are columns in the data DataFrame
        y_name (string): name of the outcome variable

        Returns
        ----------
        (pd.DataFrame): a data frame containing the feature importance statistics
        """
        all_result = pd.DataFrame()
        for x_name_i in features: 
            one_result = self._filter_F_one_feature(data=data,
                treatment_indicator=treatment_indicator, feature_name=x_name_i, y_name=y_name
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by='score', ascending=False)
        all_result['rank'] = all_result['score'].rank(ascending=False)

        return all_result


    @staticmethod
    def _filter_LR_one_feature(data, treatment_indicator, feature_name, y_name, disp=True):  
        """
        Conduct LR (Likelihood Ratio) test of the interaction between treatment and one feature.

        Parameters
        ----------
        data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
        treatment_indicator (string): the column name for binary indicator of treatment (value 1) or control (value 0) 
        feature_name (string): feature name, as one column in the data DataFrame
        y_name (string): name of the outcome variable

        Returns
        ----------
        (pd.DataFrame): a data frame containing the feature importance statistics
        """
        Y = data[y_name]
        
        # Restricted model
        X_r = data[[treatment_indicator, feature_name]]
        X_r = sm.add_constant(X_r)
        model_r = sm.Logit(Y, X_r)
        result_r = model_r.fit(disp=disp)

        # Full model (with interaction)
        X_f = X_r.copy()
        X_f['{}-{}'.format(treatment_indicator, feature_name)] = X_f[[treatment_indicator, feature_name]].product(axis=1)
        model_f = sm.Logit(Y, X_f)
        result_f = model_f.fit(disp=disp)

        LR_stat = -2 * (result_r.llf - result_f.llf)
        LR_df = len(result_f.params) - len(result_r.params)
        LR_pvalue = 1 - stats.chi2.cdf(LR_stat, df=LR_df)

        LR_test_result = pd.DataFrame({
            'feature': feature_name, # for the interaction, not the main effect
            'method': 'LRT-statistic',
            'score': LR_stat, 
            'p_value': LR_pvalue,
            'misc': 'df: {}'.format(LR_df), 
        }, index=[0]).reset_index(drop=True)

        return LR_test_result


    def filter_LR(self, data, treatment_indicator, features, y_name, disp=True):
        """
        Rank features based on the LRT-statistics of the interaction.

        Parameters
        ----------
        data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
        treatment_indicator (string): the column name for binary indicator of treatment (value 1) or control (value 0) 
        feature_name (string): feature name, as one column in the data DataFrame
        y_name (string): name of the outcome variable

        Returns
        ----------
        (pd.DataFrame): a data frame containing the feature importance statistics
        """
        all_result = pd.DataFrame()
        for x_name_i in features: 
            one_result = self._filter_LR_one_feature(data=data, 
                treatment_indicator=treatment_indicator, feature_name=x_name_i, y_name=y_name, disp=disp
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by='score', ascending=False)
        all_result['rank'] = all_result['score'].rank(ascending=False)

        return all_result


    # Get node summary - a function 
    @staticmethod
    def _GetNodeSummary(data,
                        experiment_group_column='treatment_group_key', 
                        y_name='conversion', smooth=True):
        """
        To count the conversions and get the probabilities by treatment groups. This function comes from the uplift tree algorithm, that is used for tree node split evaluation.

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
                    results[ti].update({ci: results_series[ti, ci]
                                        if results_series.index.isin([(ti, ci)]).any()
                                        else 1})
                else:
                    results[ti].update({ci: results_series[ti, ci]})

        # Probability of conversion and group size by treatment group
        nodeSummary = {}
        for treatment_group_key in results: 
            n_1 = results[treatment_group_key][1]
            n_total = (results[treatment_group_key][1] 
                       + results[treatment_group_key][0])
            y_mean = 1.0 * n_1 / n_total
            nodeSummary[treatment_group_key] = [y_mean, n_total]
        
        return results, nodeSummary 

    # Divergence-related functions, from upliftpy
    @staticmethod
    def _kl_divergence(pk, qk):
        """
        Calculate KL Divergence for binary classification.

        Parameters
        ----------
        pk (float): Probability of class 1 in treatment group
        qk (float): Probability of class 1 in control group
        """
        if qk < 0.1**6:
            qk = 0.1**6
        elif qk > 1 - 0.1**6:
            qk = 1 - 0.1**6
        S = pk * np.log(pk / qk) + (1-pk) * np.log((1-pk) / (1-qk))
        return S

    def _evaluate_KL(self, nodeSummary, control_group='control'):
        """
        Calculate the multi-treatment unconditional D (one node)
        with KL Divergence as split Evaluation function.

        Parameters
        ----------
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
    def _evaluate_ED(nodeSummary, control_group='control'):
        """
        Calculate the multi-treatment unconditional D (one node)
        with Euclidean Distance as split Evaluation function.

        Parameters
        ----------
        nodeSummary (dict): a dictionary containing the statistics for a tree node sample
        control_group (string, optional, default='control'): the name for control group        
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += 2 * (nodeSummary[treatment_group][0] - pc)**2
        return d_res

    @staticmethod
    def _evaluate_Chi(nodeSummary, control_group='control'):
        """
        Calculate the multi-treatment unconditional D (one node)
        with Chi-Square as split Evaluation function.

        Parameters
        ----------
        nodeSummary (dict): a dictionary containing the statistics for a tree node sample
        control_group (string, optional, default='control'): the name for control group 

        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += (
                    (nodeSummary[treatment_group][0] - pc)**2 / max(0.1**6, pc) 
                    + (nodeSummary[treatment_group][0] - pc)**2 / max(0.1**6, 1-pc)
                )
        return d_res


    def _filter_D_one_feature(self, data, feature_name, y_name, 
                              n_bins=10, method='KL', control_group='control',
                              experiment_group_column='treatment_group_key'):
        """
        Calculate the chosen divergence measure for one feature.

        Parameters
        ----------
        data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
        treatment_indicator (string): the column name for binary indicator of treatment (value 1) or control (value 0) 
        feature_name (string): feature name, as one column in the data DataFrame
        y_name (string): name of the outcome variable
        method (string, optional, default = 'KL'): taking one of the following values {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.  
                'F' for F-test 
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods, KL divergence, Euclidean distance, Chi-Square respectively
        experiment_group_column (string, optional, default = 'treatment_group_key'): the experiment column name in the DataFrame, which contains the treatment and control assignment label
        control_group (string, optional, default = 'control'): name for control group, value in the experiment group column
        n_bins (int, optional, default = 10): number of bins to be used for bin-based uplift filter methods
        Returns
        ----------
        (pd.DataFrame): a data frame containing the feature importance statistics
        """
        # [TODO] Application to categorical features

        if method == 'KL':
            evaluationFunction = self._evaluate_KL
        elif method == 'ED':
            evaluationFunction = self._evaluate_ED
        elif method == 'Chi':
            evaluationFunction = self._evaluate_Chi

        totalSize = len(data.index)

        # drop duplicate edges in pq.cut result to avoid issues
        x_bin = pd.qcut(data[feature_name].values, n_bins, labels=False, 
                        duplicates='drop')
        d_children = 0
        for i_bin in range(x_bin.max() + 1): # range(n_bins):
            nodeSummary = self._GetNodeSummary(
                data=data.loc[x_bin == i_bin], 
                experiment_group_column=experiment_group_column, y_name=y_name
            )[1]
            nodeScore = evaluationFunction(nodeSummary, 
                                           control_group=control_group)
            nodeSize = sum([x[1] for x in list(nodeSummary.values())])
            d_children += nodeScore * nodeSize / totalSize

        parentNodeSummary = self._GetNodeSummary(
            data=data, experiment_group_column=experiment_group_column, y_name=y_name
        )[1]
        d_parent = evaluationFunction(parentNodeSummary, 
                                      control_group=control_group)
            
        d_res = d_children - d_parent
        
        D_result = pd.DataFrame({
            'feature': feature_name, 
            'method': method,
            'score': d_res, 
            'p_value': None,
            'misc': 'number_of_bins: {}'.format(min(n_bins, x_bin.max()+1)),# format(n_bins),
        }, index=[0]).reset_index(drop=True)

        return(D_result)

    def filter_D(self, data, features, y_name, 
                 n_bins=10, method='KL', control_group='control',
                 experiment_group_column='treatment_group_key'):
        """
        Rank features based on the chosen divergence measure.

        Parameters
        ----------
        data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
        treatment_indicator (string): the column name for binary indicator of treatment (value 1) or control (value 0) 
        features (list of string): list of feature names, that are columns in the data DataFrame
        y_name (string): name of the outcome variable
        method (string, optional, default = 'KL'): taking one of the following values {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.  
                'F' for F-test 
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods, KL divergence, Euclidean distance, Chi-Square respectively
        experiment_group_column (string, optional, default = 'treatment_group_key'): the experiment column name in the DataFrame, which contains the treatment and control assignment label
        control_group (string, optional, default = 'control'): name for control group, value in the experiment group column
        n_bins (int, optional, default = 10): number of bins to be used for bin-based uplift filter methods

        Returns
        ----------
        (pd.DataFrame): a data frame containing the feature importance statistics
        """
        
        all_result = pd.DataFrame()

        for x_name_i in features: 
            one_result = self._filter_D_one_feature(
                data=data, feature_name=x_name_i, y_name=y_name,
                n_bins=n_bins, method=method, control_group=control_group,
                experiment_group_column=experiment_group_column, 
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by='score', ascending=False)
        all_result['rank'] = all_result['score'].rank(ascending=False)

        return all_result

    def get_importance(self, data, features, y_name, method, 
                      experiment_group_column='treatment_group_key',
                      control_group = 'control', 
                      treatment_group = 'treatment',
                      n_bins=5, 
                      ):
        """
        Rank features based on the chosen statistic of the interaction.

        Parameters
        ----------
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            features (list of string): list of feature names, that are columns in the data DataFrame
            y_name (string): name of the outcome variable
            method (string, optional, default = 'KL'): taking one of the following values {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.  
                'F' for F-test 
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods, KL divergence, Euclidean distance, Chi-Square respectively
            experiment_group_column (string): the experiment column name in the DataFrame, which contains the treatment and control assignment label
            control_group (string): name for control group, value in the experiment group column
            treatment_group (string): name for treatment group, value in the experiment group column
            n_bins (int, optional): number of bins to be used for bin-based uplift filter methods
        
        Returns
        ----------
            (pd.DataFrame): a data frame with following columns: ['method', 'feature', 'rank', 'score', 'p_value', 'misc']
        """
        
        if method == 'F':
            data = data[data[experiment_group_column].isin([control_group, treatment_group])]
            data['treatment_indicator'] = 0
            data.loc[data[experiment_group_column]==treatment_group,'treatment_indicator'] = 1
            all_result = self.filter_F(data=data, 
                treatment_indicator='treatment_indicator', features=features, y_name=y_name
            )
        elif method == 'LR':
            data = data[data[experiment_group_column].isin([control_group, treatment_group])]
            data['treatment_indicator'] = 0
            data.loc[data[experiment_group_column]==treatment_group,'treatment_indicator'] = 1
            all_result = self.filter_LR(data=data, disp=True,
                treatment_indicator='treatment_indicator', features=features, y_name=y_name
            )
        else:
            all_result = self.filter_D(data=data, method=method,
                features=features, y_name=y_name, 
                n_bins=n_bins, control_group=control_group,
                experiment_group_column=experiment_group_column, 
            )
        
        all_result['method'] = method + ' filter'
        return all_result[['method', 'feature', 'rank', 'score', 'p_value', 'misc']]
