import numpy as np
import pandas as pd
import pytest
from causalml.features import OneHotEncoder, LabelEncoder, load_data


@pytest.fixture
def generate_categorical_data():
    generated = False

    def _generate_data():
        if not generated:
            df = pd.DataFrame(
                {
                    "cat1": ["a", "a", "b", "a", "c", "b", "d"],
                    "cat2": ["aa", "aa", "aa", "bb", "bb", "bb", "cc"],
                    "num1": [1, 2, 1, 2, 1, 1, 1],
                }
            )

        return df

    yield _generate_data


def test_load_data(generate_categorical_data):
    df = generate_categorical_data()

    features = load_data(df, df.columns)

    assert df.shape[0] == features.shape[0]


def test_load_data_all_numeric_features():
    # OneHotEncoder asserts when nothing is categorical, so load_data used to
    # raise instead of returning the numeric columns unchanged.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]})

    features = load_data(df, df.columns)

    assert isinstance(features, np.ndarray) and not isinstance(features, np.matrix)
    assert features.shape == (3, 2)
    np.testing.assert_array_equal(features, df.values)


def test_load_data_categorical_column_that_encodes_to_nothing():
    # A single-valued categorical column maps entirely to label 0, which
    # _transform_col drops, so the encoder produces nothing and used to assert.
    df = pd.DataFrame({"const_cat": ["x"] * 7, "num1": [1, 2, 1, 2, 1, 1, 1]})

    features = load_data(df, ["const_cat", "num1"])

    assert isinstance(features, np.ndarray) and not isinstance(features, np.matrix)
    assert features.shape == (7, 1)
    np.testing.assert_array_equal(features, df[["num1"]].values)


def test_load_data_when_transform_returns_none(monkeypatch):
    # Under python -O the assertion inside OneHotEncoder.transform is stripped
    # and it returns None rather than raising, so load_data has to handle the
    # value as well as the exception.
    monkeypatch.setattr(OneHotEncoder, "fit_transform", lambda self, X, y=None: None)
    df = pd.DataFrame({"const_cat": ["x"] * 7, "num1": [1, 2, 1, 2, 1, 1, 1]})

    features = load_data(df, ["const_cat", "num1"])

    assert isinstance(features, np.ndarray) and not isinstance(features, np.matrix)
    assert features.shape == (7, 1)
    np.testing.assert_array_equal(features, df[["num1"]].values)


def test_load_data_returns_ndarray_with_categorical_features(generate_categorical_data):
    df = generate_categorical_data()

    features = load_data(df, df.columns)

    assert isinstance(features, np.ndarray) and not isinstance(features, np.matrix)


def test_LabelEncoder(generate_categorical_data):
    df = generate_categorical_data()
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    n_category = 0
    for col in cat_cols:
        n_category += df[col].nunique()

    lbe = LabelEncoder(min_obs=2)
    X_cat = lbe.fit_transform(df[cat_cols])
    n_label = 0
    for col in cat_cols:
        n_label += X_cat[col].nunique()

    assert df.shape[0] == X_cat.shape[0] and n_label < n_category


def test_OneHotEncoder(generate_categorical_data):
    df = generate_categorical_data()
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    n_category = 0
    for col in cat_cols:
        n_category += df[col].nunique()

    ohe = OneHotEncoder(min_obs=2)
    X_cat = ohe.fit_transform(df[cat_cols]).todense()

    assert df.shape[0] == X_cat.shape[0] and X_cat.shape[1] < n_category
