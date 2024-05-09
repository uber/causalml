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
    learner.fit(X_train, treatment_train, y_train)
    cate_test = learner.predict(X_test, treatment_test).flatten()

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
        calibrate_propensity=True,
        ci=False,
    )
