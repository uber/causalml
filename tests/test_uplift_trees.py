import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import get_cumgain

from .const import RANDOM_SEED, N_SAMPLE


CONTROL_NAME = 'c'
TREATMENT_NAMES = [CONTROL_NAME, 'treatment1', 'treatment2', 'treatment3']
CONVERSION = 'conversion'


@pytest.fixture
def generate_data():

    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(n_samples=N_SAMPLE,
                                              treatment_name=TREATMENT_NAMES,
                                              y_name=CONVERSION,
                                              random_seed=RANDOM_SEED)

        return data

    yield _generate_data


def test_make_uplift_classification(generate_data):
    df, x_names = generate_data()
    assert df.shape[0] == N_SAMPLE * len(TREATMENT_NAMES)


def test_UpliftRandomForestClassifier(generate_data):
    df, x_names = generate_data()
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=RANDOM_SEED)

    # Train the UpLift Random Forest classifer
    uplift_model = UpliftRandomForestClassifier(
        control_name=TREATMENT_NAMES[0]
    )

    uplift_model.fit(df_train[x_names].values,
                     treatment=df_train['treatment_group_key'].values,
                     y=df_train[CONVERSION].values)

    y_pred = uplift_model.predict(df_test[x_names].values)
    result = pd.DataFrame(y_pred, columns=uplift_model.classes_)

    best_treatment = np.where((result < 0).all(axis=1),
                              CONTROL_NAME,
                              result.idxmax(axis=1))

    # Create a synthetic population:

    # Create indicator variables for whether a unit happened to have the
    # recommended treatment or was in the control group
    actual_is_best = np.where(
        df_test['treatment_group_key'] == best_treatment, 1, 0
    )
    actual_is_control = np.where(
        df_test['treatment_group_key'] == CONTROL_NAME, 1, 0
    )

    synthetic = (actual_is_best == 1) | (actual_is_control == 1)
    synth = result[synthetic]

    auuc_metrics = synth.assign(
        is_treated=1 - actual_is_control[synthetic],
        conversion=df_test.loc[synthetic, CONVERSION].values,
        uplift_tree=synth.max(axis=1)
    ).drop(columns=list(uplift_model.classes_))

    cumgain = get_cumgain(auuc_metrics,
                          outcome_col=CONVERSION,
                          treatment_col='is_treated')

    # Check if the cumulative gain of UpLift Random Forest is higher than
    # random
    assert cumgain['uplift_tree'].sum() > cumgain['Random'].sum()


def test_UpliftTreeClassifier(generate_data):
    df, x_names = generate_data()
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=RANDOM_SEED)

    # Train the UpLift Random Forest classifer
    uplift_model = UpliftTreeClassifier(control_name=TREATMENT_NAMES[0])

    uplift_model.fit(df_train[x_names].values,
                     treatment=df_train['treatment_group_key'].values,
                     y=df_train[CONVERSION].values)

    _, _, _, y_pred = uplift_model.predict(df_test[x_names].values,
                                           full_output=True)
    result = pd.DataFrame(y_pred)
    result.drop(CONTROL_NAME, axis=1, inplace=True)

    best_treatment = np.where((result < 0).all(axis=1),
                              CONTROL_NAME,
                              result.idxmax(axis=1))

    # Create a synthetic population:

    # Create indicator variables for whether a unit happened to have the
    # recommended treatment or was in the control group
    actual_is_best = np.where(
        df_test['treatment_group_key'] == best_treatment, 1, 0
    )
    actual_is_control = np.where(
        df_test['treatment_group_key'] == CONTROL_NAME, 1, 0
    )

    synthetic = (actual_is_best == 1) | (actual_is_control == 1)
    synth = result[synthetic]

    auuc_metrics = synth.assign(
        is_treated=1 - actual_is_control[synthetic],
        conversion=df_test.loc[synthetic, CONVERSION].values,
        uplift_tree=synth.max(axis=1)
    ).drop(columns=result.columns)

    cumgain = get_cumgain(auuc_metrics,
                          outcome_col=CONVERSION,
                          treatment_col='is_treated')

    # Check if the cumulative gain of UpLift Random Forest is higher than
    # random
    assert cumgain['uplift_tree'].sum() > cumgain['Random'].sum()
