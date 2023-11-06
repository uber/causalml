import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from causalml.dataset import make_uplift_classification
from causalml.inference.meta import BaseTClassifier
from causalml.optimize.value_optimization import CounterfactualValueEstimator
from causalml.optimize.utils import get_treatment_costs
from causalml.optimize.utils import get_actual_value


from tests.const import RANDOM_SEED


def test_counterfactual_value_optimization():
    df, X_names = make_uplift_classification(
        n_samples=2000, treatment_name=["control", "treatment1", "treatment2"]
    )
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    train_idx = df_train.index
    test_idx = df_test.index

    conversion_cost_dict = {"control": 0, "treatment1": 2.5, "treatment2": 5}
    impression_cost_dict = {"control": 0, "treatment1": 0, "treatment2": 0.02}

    cc_array, ic_array, conditions = get_treatment_costs(
        treatment=df["treatment_group_key"],
        control_name="control",
        cc_dict=conversion_cost_dict,
        ic_dict=impression_cost_dict,
    )
    conversion_value_array = np.full(df.shape[0], 20)

    actual_value = get_actual_value(
        treatment=df["treatment_group_key"],
        observed_outcome=df["conversion"],
        conversion_value=conversion_value_array,
        conditions=conditions,
        conversion_cost=cc_array,
        impression_cost=ic_array,
    )

    random_allocation_value = actual_value.loc[test_idx].mean()

    tm = BaseTClassifier(learner=LogisticRegression(), control_name="control")
    tm.fit(
        df_train[X_names].values,
        df_train["treatment_group_key"],
        df_train["conversion"],
    )
    tm_pred = tm.predict(df_test[X_names].values)

    proba_model = LogisticRegression()

    W_dummies = pd.get_dummies(df["treatment_group_key"])
    XW = np.c_[df[X_names], W_dummies]
    proba_model.fit(XW[train_idx], df_train["conversion"])
    y_proba = proba_model.predict_proba(XW[test_idx])[:, 1]

    cve = CounterfactualValueEstimator(
        treatment=df_test["treatment_group_key"],
        control_name="control",
        treatment_names=conditions[1:],
        y_proba=y_proba,
        cate=tm_pred,
        value=conversion_value_array[test_idx],
        conversion_cost=cc_array[test_idx],
        impression_cost=ic_array[test_idx],
    )

    cve_best_idx = cve.predict_best()
    cve_best = [conditions[idx] for idx in cve_best_idx]
    actual_is_cve_best = df.loc[test_idx, "treatment_group_key"] == cve_best
    cve_value = actual_value.loc[test_idx][actual_is_cve_best].mean()

    assert cve_value > random_allocation_value
