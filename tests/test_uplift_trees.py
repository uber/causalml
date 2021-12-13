import cProfile
import numpy as np
import pandas as pd
import pstats
from sklearn.model_selection import train_test_split

from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
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
        control_name=TREATMENT_NAMES[0],
        random_state=RANDOM_SEED
    )

    uplift_model.fit(df_train[x_names].values,
                     treatment=df_train['treatment_group_key'].values,
                     y=df_train[CONVERSION].values)

    y_pred = uplift_model.predict(df_test[x_names].values)
    result = pd.DataFrame(y_pred, columns=uplift_model.classes_[1:])

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
    ).drop(columns=list(uplift_model.classes_[1:]))

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
    uplift_model = UpliftTreeClassifier(control_name=TREATMENT_NAMES[0], random_state=RANDOM_SEED)

    pr = cProfile.Profile(subcalls=True, builtins=True, timeunit=.001)
    pr.enable()
    uplift_model.fit(df_train[x_names].values,
                     treatment=df_train['treatment_group_key'].values,
                     y=df_train[CONVERSION].values)

    y_pred = uplift_model.predict(df_test[x_names].values)
    pr.disable()
    with open('UpliftTreeClassifier.prof', 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
        ps.print_stats()

    result = pd.DataFrame(y_pred, columns=uplift_model.classes_)
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


def test_UpliftTreeClassifier_feature_importance(generate_classification_data):
    # test if feature importance is working as expected
    df, x_names = generate_classification_data()
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=RANDOM_SEED)

    # Train the upLift classifier
    uplift_model = UpliftTreeClassifier(control_name=TREATMENT_NAMES[0], random_state=RANDOM_SEED)
    uplift_model.fit(df_train[x_names].values,
                     treatment=df_train['treatment_group_key'].values,
                     y=df_train[CONVERSION].values)

    assert hasattr(uplift_model, 'feature_importances_')
    assert (np.all(uplift_model.feature_importances_ >= 0))
    num_non_zero_imp_features = sum([1 if imp > 0 else 0 for imp in uplift_model.feature_importances_])

    def getNonleafCount(node):
        # base case
        if (node is None or (node.trueBranch is None and
                             node.falseBranch is None)):
            return 0
        # If root is Not None and its one of its child is also not None
        return (1 + getNonleafCount(node.trueBranch) +
                getNonleafCount(node.falseBranch))

    num_non_leaf_nodes = getNonleafCount(uplift_model.fitted_uplift_tree)
    # Check if the features with positive importance is not more than number of nodes
    # the reason is, each non-leaf node evaluates only one feature, and some of the nodes
    # would evaluate the same feature, thus the number of features with importance value
    # shouldn't be larger than the number of non-leaf node
    assert num_non_zero_imp_features <= num_non_leaf_nodes
