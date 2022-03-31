import pytest

from causalml.dataset import (
    simulate_nuisance_and_easy_treatment,
    simulate_hidden_confounder,
    simulate_randomized_trial,
)
from causalml.dataset import (
    get_synthetic_preds,
    get_synthetic_summary,
    get_synthetic_auuc,
)
from causalml.dataset import get_synthetic_preds_holdout, get_synthetic_summary_holdout
from causalml.inference.meta import LRSRegressor, XGBTRegressor


@pytest.mark.parametrize(
    "synthetic_data_func",
    [
        simulate_nuisance_and_easy_treatment,
        simulate_hidden_confounder,
        simulate_randomized_trial,
    ],
)
def test_get_synthetic_preds(synthetic_data_func):
    preds_dict = get_synthetic_preds(
        synthetic_data_func=synthetic_data_func,
        n=1000,
        estimators={
            "S Learner (LR)": LRSRegressor(),
            "T Learner (XGB)": XGBTRegressor(),
        },
    )

    assert (
        preds_dict["S Learner (LR)"].shape[0] == preds_dict["T Learner (XGB)"].shape[0]
    )


def test_get_synthetic_summary():
    summary = get_synthetic_summary(
        synthetic_data_func=simulate_nuisance_and_easy_treatment,
        estimators={
            "S Learner (LR)": LRSRegressor(),
            "T Learner (XGB)": XGBTRegressor(),
        },
    )

    print(summary)


def test_get_synthetic_preds_holdout():
    preds_train, preds_valid = get_synthetic_preds_holdout(
        synthetic_data_func=simulate_nuisance_and_easy_treatment,
        n=1000,
        estimators={
            "S Learner (LR)": LRSRegressor(),
            "T Learner (XGB)": XGBTRegressor(),
        },
    )

    assert (
        preds_train["S Learner (LR)"].shape[0]
        == preds_train["T Learner (XGB)"].shape[0]
    )
    assert (
        preds_valid["S Learner (LR)"].shape[0]
        == preds_valid["T Learner (XGB)"].shape[0]
    )


def test_get_synthetic_summary_holdout():
    summary = get_synthetic_summary_holdout(
        synthetic_data_func=simulate_nuisance_and_easy_treatment
    )

    print(summary)


def test_get_synthetic_auuc():
    preds_dict = get_synthetic_preds(
        synthetic_data_func=simulate_nuisance_and_easy_treatment,
        n=1000,
        estimators={
            "S Learner (LR)": LRSRegressor(),
            "T Learner (XGB)": XGBTRegressor(),
        },
    )

    auuc_df = get_synthetic_auuc(preds_dict, plot=False)
    print(auuc_df)
