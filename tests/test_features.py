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


def test_LabelEncoder(generate_categorical_data):
    df = generate_categorical_data()
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
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
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    n_category = 0
    for col in cat_cols:
        n_category += df[col].nunique()

    ohe = OneHotEncoder(min_obs=2)
    X_cat = ohe.fit_transform(df[cat_cols]).todense()

    assert df.shape[0] == X_cat.shape[0] and X_cat.shape[1] < n_category
