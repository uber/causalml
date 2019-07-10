from __future__ import absolute_import
import logging
from kaggler.preprocessing import OneHotEncoder
import numpy as np


logger = logging.getLogger('causalml')


def transform_features(df, features, transformations={}):
    """Transform features.

    Args:
        df (pandas.DataFrame): an input data frame
        features (list of str): column names to be used in the inference model
        transformations (dict of (str, func)): transformations to be applied to features

    Returns:
        (numpy.matrix): a feature matrix
    """

    df = df[features].copy()

    bool_cols = [col for col in df.columns if df[col].dtype == bool]
    df.loc[:, bool_cols] = df[bool_cols].astype(np.int8)

    for col, transformation in transformations.items():
        logger.info('Applying {} to {}'.format(transformation.__name__, col))
        df[col] = df[col].apply(transformation)

    cat_cols = [col for col in features if df[col].dtype == np.object]
    num_cols = [col for col in features if col not in cat_cols]

    logger.info('Applying one-hot-encoding to {}'.format(cat_cols))
    ohe = OneHotEncoder(min_obs=df.shape[0] * 0.001)
    X_cat = ohe.fit_transform(df[cat_cols]).todense()

    X = np.hstack([df[num_cols].values, X_cat])

    return X
