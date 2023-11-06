import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from causalml.dataset import make_uplift_classification
from causalml.optimize.unit_selection import CounterfactualUnitSelector
from causalml.optimize.utils import get_treatment_costs
from causalml.optimize.utils import get_actual_value

from tests.const import RANDOM_SEED


def test_counterfactual_unit_selection():
    df, X_names = make_uplift_classification(
        n_samples=2000, treatment_name=["control", "treatment"]
    )
    df["treatment_numeric"] = df["treatment_group_key"].replace(
        {"control": 0, "treatment": 1}
    )
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    train_idx = df_train.index
    test_idx = df_test.index

    conversion_cost_dict = {"control": 0, "treatment": 2.5}
    impression_cost_dict = {"control": 0, "treatment": 0}

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

    nevertaker_payoff = 0
    alwaystaker_payoff = -2.5
    complier_payoff = 17.5
    defier_payoff = -20

    cus = CounterfactualUnitSelector(
        learner=LogisticRegressionCV(),
        nevertaker_payoff=nevertaker_payoff,
        alwaystaker_payoff=alwaystaker_payoff,
        complier_payoff=complier_payoff,
        defier_payoff=defier_payoff,
    )

    cus.fit(
        data=df_train.drop("treatment_group_key", axis=1),
        treatment="treatment_numeric",
        outcome="conversion",
    )

    cus_pred = cus.predict(
        data=df_test.drop("treatment_group_key", axis=1),
        treatment="treatment_numeric",
        outcome="conversion",
    )

    best_cus = np.where(cus_pred > 0, 1, 0)
    actual_is_cus = df_test["treatment_numeric"] == best_cus.ravel()
    cus_value = actual_value.loc[test_idx][actual_is_cus].mean()

    assert cus_value > random_allocation_value
