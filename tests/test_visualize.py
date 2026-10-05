from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split

from causalml.metrics.visualize import get_cumlift, plot_tmlegain
from causalml.inference.meta import LRSRegressor


def test_visualize_get_cumlift_errors_on_nan():
    df = pd.DataFrame(
        [[0, np.nan, 0.5], [1, np.nan, 0.1], [1, 1, 0.4], [0, 1, 0.3], [1, 1, 0.2]],
        columns=["w", "y", "pred"],
    )

    with pytest.raises(Exception):
        get_cumlift(df)


def test_plot_tmlegain(generate_regression_data, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    y, X, treatment, tau, b, e = generate_regression_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        e_train,
        e_test,
        treatment_train,
        treatment_test,
        tau_train,
        tau_test,
        b_train,
        b_test,
    ) = train_test_split(X, y, e, treatment, tau, b, test_size=0.5, random_state=42)

    learner = LRSRegressor()
    learner.fit(X=X_train, treatment=treatment_train, y=y_train)
    cate_test = learner.predict(X=X_test, treatment=treatment_test).flatten()

    df = pd.DataFrame(
        {
            "y": y_test,
            "w": treatment_test,
            "p": e_test,
            "S-Learner": cate_test,
            "Actual": tau_test,
        }
    )

    inference_cols = []
    for i in range(X_test.shape[1]):
        col = "col_" + str(i)
        df[col] = X_test[:, i]
        inference_cols.append(col)

    n_fold = 3
    kf = KFold(n_splits=n_fold)

    plot_tmlegain(
        df,
        inference_col=inference_cols,
        outcome_col="y",
        treatment_col="w",
        p_col="p",
        n_segment=5,
        cv=kf,
        ci=False,
    )


def test_get_std_diffs_skips_non_numeric_covariates():
    """A string covariate has no mean, so it belongs in the dropped set.

    Before, any column with enough distinct values was classified as
    continuous whatever its dtype, and the diagnostic died inside numpy with
    "ufunc 'divide' not supported for the input types" rather than reporting
    the balance of the columns it could measure.
    """
    from causalml.metrics.visualize import _get_numeric_vars, get_std_diffs

    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame(
        {
            "age": rng.normal(size=n),
            "group": rng.choice(list("abcdef"), size=n),
            "flag": rng.randint(0, 2, size=n),
        }
    )
    w = pd.Series(rng.randint(0, 2, size=n))

    cont_cols, prop_cols = _get_numeric_vars(X)
    assert cont_cols == ["age"]
    assert prop_cols == ["flag"]

    std_diffs = get_std_diffs(X, w)
    assert list(std_diffs.index) == ["age", "flag"]
    assert np.isfinite(std_diffs.values).all()


def test_get_std_diffs_errors_when_no_covariate_is_usable():
    from causalml.metrics.visualize import get_std_diffs

    rng = np.random.RandomState(42)
    n = 20
    X = pd.DataFrame({"group": rng.choice(list("abcdef"), size=n)})
    w = pd.Series(rng.randint(0, 2, size=n))

    with pytest.raises(ValueError):
        get_std_diffs(X, w)
