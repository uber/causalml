from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


logger = logging.getLogger('causalml')


def smd(feature, treatment):
    """Calculate the standard mean difference (SMD) of a feature between the treatment and control groups.

    The definition is available at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/#s11title

    Args:
        feature (pandas.Series): a column of a feature to calculate SMD for
        treatment (pandas.Series): a column that indicate whether a row is in the treatment group or not

    Returns:
        (float): The SMD of the feature
    """
    t = feature[treatment == 1]
    c = feature[treatment == 0]
    return (t.mean() - c.mean()) / np.sqrt(.5 * (t.var() + c.var()))


def create_table_one(data, treatment_col, features):
    """Report balance in input features between the treatment and control groups.

    References:
        R's tableone at CRAN: https://github.com/kaz-yos/tableone
        Python's tableone at PyPi: https://github.com/tompollard/tableone

    Args:
        data (pandas.DataFrame): total or matched sample data
        treatmenet_col (str): the column name for the treatment
        features (list of str): the column names of features

    Returns:
        (pandas.DataFrame): A table with the means and standard deviations in the treatment and control groups,
        and the SMD between two groups for the features.
    """
    t1 = pd.pivot_table(data[features + [treatment_col]],
                        columns=treatment_col,
                        aggfunc=[lambda x: '{:.2f} ({:.2f})'.format(x.mean(), x.std())])
    t1.columns = t1.columns.droplevel(level = 0)
    t1['SMD'] = data[features].apply(lambda x: smd(x, data[treatment_col])).round(4)

    n_row = pd.pivot_table(data[[features[0], treatment_col]],
                           columns=treatment_col,
                           aggfunc=['count'])
    n_row.columns = n_row.columns.droplevel(level = 0)
    n_row['SMD'] = ''
    n_row.index = ['n']

    t1 = pd.concat([n_row, t1], axis = 0)
    t1.columns.name = ''
    t1.columns = ['Control', 'Treatment', 'SMD']
    t1.index.name = 'Variable'

    return t1


class NearestNeighborMatch(object):
    """
    Propensity score matching based on the nearest neighbor algorithm.

    Attributes:
        caliper (float): threshold to be considered as a match.
        replace (bool): whether to match with replacement or not
        ratio (int): ratio of control / treatment to be matched. used only if replace=True.
        shuffle (bool): whether to shuffle the treatment group data before matching
        random_state (numpy.random.RandomState or int): RandomState or an int seed
    """

    def __init__(self, caliper=.2, replace=False, ratio=1, shuffle=True, random_state=None):
        """Initialize a propensity score matching model.

        Args:
            caliper (float): threshold to be considered as a match.
            replace (bool): whether to match with replacement or not
            shuffle (bool): whether to shuffle the treatment group data before matching or not
            random_state (numpy.random.RandomState or int): RandomState or an int seed
        """
        self.caliper = caliper
        self.replace = replace
        self.ratio = ratio
        self.shuffle = shuffle
        self.random_state = check_random_state(random_state)

    def match(self, data, treatment_col, score_cols):
        """
        Find matches from the control group by matching on specified columns (propensity preferred).

        Args:
            data (pandas.DataFrame): total input data
            treatment_col (str): the column name for the treatment
            score_cols (list): list of column names for matching (propensity column should be included)

        Returns:
            (pandas.DataFrame): The subset of data consisting of matched treatment and control group data.
        """
        assert type(score_cols) == list, 'score_cols must be a list'
        treatment = data.loc[data[treatment_col] == 1, score_cols]
        control = data.loc[data[treatment_col] == 0, score_cols]

        sdcal = self.caliper * np.std(data[score_cols].values)

        if self.replace:
            matching_model = NearestNeighbors(n_neighbors=self.ratio)
            matching_model.fit(control)
            distances, indices = matching_model.kneighbors(treatment)

            # distances and indices are (n_obs, self.ratio) matrices.
            # To index easily, reshape distances, indices and treatment into the (n_obs * self.ratio, 1)
            # matrices and data frame.
            distances = distances.T.flatten()
            indices = indices.T.flatten()
            treatment = pd.concat([treatment] * self.ratio, axis=0)

            cond = distances < sdcal
            t_idx_matched = list(set(treatment.loc[cond].index.tolist()))   # Deduplicate the indices of the treatment gruop
            c_idx_matched = control.iloc[indices[cond]].index.tolist()      # XXX: Should we dedulicate the indices of the control group too?
        else:
            assert len(score_cols)==1, 'Matching on multiple columns is only supported using the replacement method (if matching on multiple columns, set replace=True).'
            score_col = score_cols[0] # unpack score_cols for the single-variable matching case

            if self.shuffle:
                t_indices = self.random_state.permutation(treatment.index)
            else:
                t_indices = treatment.index

            t_idx_matched = []
            c_idx_matched = []
            control['unmatched'] = True

            for t_idx in t_indices:
                dist = np.abs(control.loc[control.unmatched, score_col] - treatment.loc[t_idx, score_col])
                c_idx_min = dist.idxmin()
                if dist[c_idx_min] <= sdcal:
                    t_idx_matched.append(t_idx)
                    c_idx_matched.append(c_idx_min)
                    control.loc[c_idx_min, 'unmatched'] = False

        return data.loc[np.concatenate([np.array(t_idx_matched),
                                        np.array(c_idx_matched)])]

    def match_by_group(self, data, treatment_col, score_cols, groupby_col):
        """
        Find matches from the control group stratified by groupby_col, by matching on specified columns (propensity preferred).

        Args:
            data (pandas.DataFrame): total sample data
            treatment_col (str): the column name for the treatment
            score_cols (list): list of column names for matching (propensity column should be included)
            groupby_col (str): the column name to be used for stratification

        Returns:
            (pandas.DataFrame): The subset of data consisting of matched treatment and control group data.
        """
        matched = data.groupby(groupby_col).apply(lambda x: self.match(data=x,
                                                                       treatment_col=treatment_col,
                                                                       score_cols=score_cols))
        return matched.reset_index(level=0, drop=True)
