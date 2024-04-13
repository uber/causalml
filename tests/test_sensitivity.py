import pandas as pd
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

from causalml.dataset import synthetic_data
from causalml.inference.meta import (
    BaseSLearner,
    BaseTLearner,
    XGBTRegressor,
    BaseXLearner,
    BaseRLearner,
)
from causalml.metrics.sensitivity import Sensitivity
from causalml.metrics.sensitivity import (
    SensitivityPlaceboTreatment,
    SensitivityRandomCause,
)
from causalml.metrics.sensitivity import (
    SensitivityRandomReplace,
    SensitivitySelectionBias,
)
from causalml.metrics.sensitivity import (
    one_sided,
    alignment,
    one_sided_att,
    alignment_att,
)

from .const import TREATMENT_COL, SCORE_COL, OUTCOME_COL, NUM_FEATURES


@pytest.mark.parametrize(
    "learner",
    [
        BaseSLearner(LinearRegression()),
        BaseTLearner(LinearRegression()),
        XGBTRegressor(),
        BaseXLearner(LinearRegression()),
        BaseRLearner(LinearRegression()),
    ],
)
def test_Sensitivity(learner):
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )

    # generate the dataset format for sensitivity analysis
    INFERENCE_FEATURES = ["feature_" + str(i) for i in range(NUM_FEATURES)]
    df = pd.DataFrame(X, columns=INFERENCE_FEATURES)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = y
    df[SCORE_COL] = e

    # calling the Base XLearner class and return the sensitivity analysis summary report
    sens = Sensitivity(
        df=df,
        inference_features=INFERENCE_FEATURES,
        p_col=SCORE_COL,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        learner=learner,
    )

    # check the sensitivity summary report
    sens_summary = sens.sensitivity_analysis(
        methods=[
            "Placebo Treatment",
            "Random Cause",
            "Subset Data",
            "Random Replace",
            "Selection Bias",
        ],
        sample_size=0.5,
    )

    print(sens_summary)


def test_SensitivityPlaceboTreatment():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )

    # generate the dataset format for sensitivity analysis
    INFERENCE_FEATURES = ["feature_" + str(i) for i in range(NUM_FEATURES)]
    df = pd.DataFrame(X, columns=INFERENCE_FEATURES)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = y
    df[SCORE_COL] = e

    # calling the Base XLearner class and return the sensitivity analysis summary report
    learner = BaseXLearner(LinearRegression())
    sens = SensitivityPlaceboTreatment(
        df=df,
        inference_features=INFERENCE_FEATURES,
        p_col=SCORE_COL,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        learner=learner,
    )

    sens_summary = sens.summary(method="Random Cause")
    print(sens_summary)


def test_SensitivityRandomCause():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )

    # generate the dataset format for sensitivity analysis
    INFERENCE_FEATURES = ["feature_" + str(i) for i in range(NUM_FEATURES)]
    df = pd.DataFrame(X, columns=INFERENCE_FEATURES)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = y
    df[SCORE_COL] = e

    # calling the Base XLearner class and return the sensitivity analysis summary report
    learner = BaseXLearner(LinearRegression())
    sens = SensitivityRandomCause(
        df=df,
        inference_features=INFERENCE_FEATURES,
        p_col=SCORE_COL,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        learner=learner,
    )

    sens_summary = sens.summary(method="Random Cause")
    print(sens_summary)


def test_SensitivityRandomReplace():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )

    # generate the dataset format for sensitivity analysis
    INFERENCE_FEATURES = ["feature_" + str(i) for i in range(NUM_FEATURES)]
    df = pd.DataFrame(X, columns=INFERENCE_FEATURES)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = y
    df[SCORE_COL] = e

    # calling the Base XLearner class and return the sensitivity analysis summary report
    learner = BaseXLearner(LinearRegression())
    sens = SensitivityRandomReplace(
        df=df,
        inference_features=INFERENCE_FEATURES,
        p_col=SCORE_COL,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        learner=learner,
        sample_size=0.9,
        replaced_feature="feature_0",
    )

    sens_summary = sens.summary(method="Random Replace")
    print(sens_summary)


def test_SensitivitySelectionBias():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )

    # generate the dataset format for sensitivity analysis
    INFERENCE_FEATURES = ["feature_" + str(i) for i in range(NUM_FEATURES)]
    df = pd.DataFrame(X, columns=INFERENCE_FEATURES)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = y
    df[SCORE_COL] = e

    # calling the Base XLearner class and return the sensitivity analysis summary report
    learner = BaseXLearner(LinearRegression())
    sens = SensitivitySelectionBias(
        df,
        INFERENCE_FEATURES,
        p_col=SCORE_COL,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        learner=learner,
        confound="alignment",
        alpha_range=None,
    )

    lls_bias_alignment, partial_rsqs_bias_alignment = sens.causalsens()
    print(lls_bias_alignment, partial_rsqs_bias_alignment)

    # Plot the results by confounding vector and plot Confidence Intervals for ATE
    sens.plot(lls_bias_alignment, ci=True)


def test_one_sided():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )
    alpha = np.quantile(y, 0.25)
    adj = one_sided(alpha, e, treatment)

    assert y.shape == adj.shape


def test_alignment():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )
    alpha = np.quantile(y, 0.25)
    adj = alignment(alpha, e, treatment)

    assert y.shape == adj.shape


def test_one_sided_att():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )
    alpha = np.quantile(y, 0.25)
    adj = one_sided_att(alpha, e, treatment)

    assert y.shape == adj.shape


def test_alignment_att():
    y, X, treatment, tau, b, e = synthetic_data(
        mode=1, n=100000, p=NUM_FEATURES, sigma=1.0
    )
    alpha = np.quantile(y, 0.25)
    adj = alignment_att(alpha, e, treatment)

    assert y.shape == adj.shape
