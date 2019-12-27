import cProfile
import numpy as np
import pandas as pd
import pstats
from sklearn.model_selection import train_test_split

from causalml.inference.tree import UpliftTreeClassifier
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import get_cumgain

from .const import RANDOM_SEED, N_SAMPLE, CONTROL_NAME, TREATMENT_NAMES, CONVERSION


def test_make_uplift_classification(generate_classification_data):
    df, _ = generate_classification_data()
    assert df.shape[0] == N_SAMPLE * len(TREATMENT_NAMES)


def test_UpliftRandomForestClassifier(generate_classification_data):
    df, x_names = generate_classification_data()
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=RANDOM_SEED)

    # Train the UpLift Random Forest classifier
    uplift_model = UpliftRandomForestClassifier(
        min_samples_leaf=50,
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


def test_UpliftTreeClassifier(generate_classification_data):
    df, x_names = generate_classification_data()
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=RANDOM_SEED)

    # Train the UpLift Random Forest classifier
    uplift_model = UpliftTreeClassifier(control_name=TREATMENT_NAMES[0])

    pr = cProfile.Profile(subcalls=True, builtins=True, timeunit=.001)
    pr.enable()
    uplift_model.fit(df_train[x_names].values,
                     treatment=df_train['treatment_group_key'].values,
                     y=df_train[CONVERSION].values)

    _, _, _, y_pred = uplift_model.predict(df_test[x_names].values,
                                           full_output=True)
    pr.disable()
    with open('UpliftTreeClassifier.prof', 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
        ps.print_stats()

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
