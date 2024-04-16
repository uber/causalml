"""
Utility functions for uplift trees.
"""

import time
from typing import Callable

import numpy as np
import pandas as pd


def cat_group(dfx, kpix, n_group=10):
    """
    Category Reduction for Categorical Variables

    Args
    ----

    dfx : dataframe
        The inputs data dataframe.

    kpix : string
        The column of the feature.

    n_group : int, optional (default = 10)
        The number of top category values to be remained, other category values will be put into "Other".

    Returns
    -------
    The transformed categorical feature value list.
    """
    if dfx[kpix].nunique() > n_group:
        # get the top categories
        top = dfx[kpix].isin(dfx[kpix].value_counts().index[:n_group])
        dfx.loc[~top, kpix] = "Other"
        return dfx[kpix].values
    else:
        return dfx[kpix].values


def cat_transform(dfx, kpix, kpi1):
    """
    Encoding string features.

    Args
    ----

    dfx : dataframe
        The inputs data dataframe.

    kpix : string
        The column of the feature.

    kpi1 : list
        The list of feature names.

    Returns
    -------
    dfx : DataFrame
        The updated dataframe containing the encoded data.

    kpi1 : list
        The updated feature names containing the new dummy feature names.
    """
    df_dummy = pd.get_dummies(dfx[kpix].values)
    new_col_names = ["%s_%s" % (kpix, x) for x in df_dummy.columns]
    df_dummy.columns = new_col_names
    dfx = pd.concat([dfx, df_dummy], axis=1)
    for new_col in new_col_names:
        if new_col not in kpi1:
            kpi1.append(new_col)
    if kpix in kpi1:
        kpi1.remove(kpix)
    return dfx, kpi1


def cv_fold_index(n, i, k, random_seed=2018):
    """
    Encoding string features.

    Args
    ----

    dfx : dataframe
        The inputs data dataframe.

    kpix : string
        The column of the feature.

    kpi1 : list
        The list of feature names.

    Returns
    -------
    dfx : DataFrame
        The updated dataframe containing the encoded data.

    kpi1 : list
        The updated feature names containing the new dummy feature names.
    """
    np.random.seed(random_seed)
    rlist = np.random.choice(a=range(k), size=n, replace=True)
    fold_i_index = np.where(rlist == i)[0]
    return fold_i_index


# Categorize continuous variable
def cat_continuous(x, granularity="Medium"):
    """
    Categorize (bin) continuous variable based on percentile.

    Args
    ----

    x : list
        Feature values.

    granularity : string, optional, (default = 'Medium')
        Control the granularity of the bins, optional values are: 'High', 'Medium', 'Low'.

    Returns
    -------
    res : list
        List of percentile bins for the feature value.
    """
    if granularity == "High":
        lspercentile = [
            np.percentile(x, 5),
            np.percentile(x, 10),
            np.percentile(x, 15),
            np.percentile(x, 20),
            np.percentile(x, 25),
            np.percentile(x, 30),
            np.percentile(x, 35),
            np.percentile(x, 40),
            np.percentile(x, 45),
            np.percentile(x, 50),
            np.percentile(x, 55),
            np.percentile(x, 60),
            np.percentile(x, 65),
            np.percentile(x, 70),
            np.percentile(x, 75),
            np.percentile(x, 80),
            np.percentile(x, 85),
            np.percentile(x, 90),
            np.percentile(x, 95),
            np.percentile(x, 99),
        ]
        res = [
            (
                "> p90 (%s)" % (lspercentile[8])
                if z > lspercentile[8]
                else (
                    "<= p10 (%s)" % (lspercentile[0])
                    if z <= lspercentile[0]
                    else (
                        "<= p20 (%s)" % (lspercentile[1])
                        if z <= lspercentile[1]
                        else (
                            "<= p30 (%s)" % (lspercentile[2])
                            if z <= lspercentile[2]
                            else (
                                "<= p40 (%s)" % (lspercentile[3])
                                if z <= lspercentile[3]
                                else (
                                    "<= p50 (%s)" % (lspercentile[4])
                                    if z <= lspercentile[4]
                                    else (
                                        "<= p60 (%s)" % (lspercentile[5])
                                        if z <= lspercentile[5]
                                        else (
                                            "<= p70 (%s)" % (lspercentile[6])
                                            if z <= lspercentile[6]
                                            else (
                                                "<= p80 (%s)" % (lspercentile[7])
                                                if z <= lspercentile[7]
                                                else (
                                                    "<= p90 (%s)" % (lspercentile[8])
                                                    if z <= lspercentile[8]
                                                    else "> p90 (%s)"
                                                    % (lspercentile[8])
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            for z in x
        ]
    elif granularity == "Medium":
        lspercentile = [
            np.percentile(x, 10),
            np.percentile(x, 20),
            np.percentile(x, 30),
            np.percentile(x, 40),
            np.percentile(x, 50),
            np.percentile(x, 60),
            np.percentile(x, 70),
            np.percentile(x, 80),
            np.percentile(x, 90),
        ]
        res = [
            (
                "<= p10 (%s)" % (lspercentile[0])
                if z <= lspercentile[0]
                else (
                    "<= p20 (%s)" % (lspercentile[1])
                    if z <= lspercentile[1]
                    else (
                        "<= p30 (%s)" % (lspercentile[2])
                        if z <= lspercentile[2]
                        else (
                            "<= p40 (%s)" % (lspercentile[3])
                            if z <= lspercentile[3]
                            else (
                                "<= p50 (%s)" % (lspercentile[4])
                                if z <= lspercentile[4]
                                else (
                                    "<= p60 (%s)" % (lspercentile[5])
                                    if z <= lspercentile[5]
                                    else (
                                        "<= p70 (%s)" % (lspercentile[6])
                                        if z <= lspercentile[6]
                                        else (
                                            "<= p80 (%s)" % (lspercentile[7])
                                            if z <= lspercentile[7]
                                            else (
                                                "<= p90 (%s)" % (lspercentile[8])
                                                if z <= lspercentile[8]
                                                else "> p90 (%s)" % (lspercentile[8])
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            for z in x
        ]
    else:
        lspercentile = [
            np.percentile(x, 15),
            np.percentile(x, 50),
            np.percentile(x, 85),
        ]
        res = [
            (
                "1-Very Low"
                if z < lspercentile[0]
                else (
                    "2-Low"
                    if z < lspercentile[1]
                    else "3-High" if z < lspercentile[2] else "4-Very High"
                )
            )
            for z in x
        ]
    return res


def kpi_transform(dfx, kpi_combo, kpi_combo_new):
    """
    Feature transformation from continuous feature to binned features for a list of features

    Args
    ----

    dfx : DataFrame
        DataFrame containing the features.

    kpi_combo : list of string
        List of feature names to be transformed

    kpi_combo_new : list of string
        List of new feature names to be assigned to the transformed features.

    Returns
    -------
    dfx : DataFrame
        Updated DataFrame containing the new features.
    """
    for j in range(len(kpi_combo)):
        if type(dfx[kpi_combo[j]].values[0]) is str:
            dfx[kpi_combo_new[j]] = dfx[kpi_combo[j]].values
            dfx[kpi_combo_new[j]] = cat_group(dfx=dfx, kpix=kpi_combo_new[j])
        else:
            if len(kpi_combo) > 1:
                dfx[kpi_combo_new[j]] = cat_continuous(
                    dfx[kpi_combo[j]].values, granularity="Low"
                )
            else:
                dfx[kpi_combo_new[j]] = cat_continuous(
                    dfx[kpi_combo[j]].values, granularity="High"
                )
    return dfx


def get_tree_leaves_mask(tree) -> np.ndarray:
    """
    Get mask array for tree leaves
    Args:
        tree: CausalTreeRegressor
              Tree object
    Returns: np.ndarray
             Mask array

    """
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]

        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    return is_leaves


def timeit(exclude_kwargs: tuple = ()) -> Callable:
    """
    timeit decorator
    Args:
        exclude_kwargs: (tuple), keyword arguments that should be excluded from display
    Returns: Callable

    """

    def wrapper(f: Callable):
        def wrapped(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            display_kw = {k: v for k, v in kw.items() if k not in exclude_kwargs}
            print(
                "Function: {} Kwargs: {} Elapsed time: {:2.4f}".format(
                    f.__name__, display_kw, te - ts
                )
            )
            return result

        return wrapped

    return wrapper
