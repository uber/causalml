import numpy as np
import pandas as pd
import pytest

from causalml.match import NearestNeighborMatch, MatchOptimizer
from causalml.propensity import ElasticNetPropensityModel
from .const import RANDOM_SEED, TREATMENT_COL, SCORE_COL, GROUP_COL


@pytest.fixture
def generate_unmatched_data(generate_regression_data):
    generated = False

    def _generate_data():
        if not generated:
            y, X, treatment, tau, b, e = generate_regression_data()

            features = ["x{}".format(i) for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=features)
            df[TREATMENT_COL] = treatment

            df_c = df.loc[treatment == 0]
            df_t = df.loc[treatment == 1]

            df = pd.concat([df_t, df_c, df_c], axis=0, ignore_index=True)

            pm = ElasticNetPropensityModel(random_state=RANDOM_SEED)
            ps = pm.fit_predict(df[features], df[TREATMENT_COL])
            df[SCORE_COL] = ps
            df[GROUP_COL] = np.random.randint(0, 2, size=df.shape[0])

        return df, features

    yield _generate_data


def test_nearest_neighbor_match_ratio_2(generate_unmatched_data):
    df, features = generate_unmatched_data()

    psm = NearestNeighborMatch(replace=False, ratio=2, random_state=RANDOM_SEED)
    matched = psm.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])
    assert sum(matched[TREATMENT_COL] == 0) == 2 * sum(matched[TREATMENT_COL] != 0)


def test_nearest_neighbor_match_by_group(generate_unmatched_data):
    df, features = generate_unmatched_data()

    psm = NearestNeighborMatch(replace=False, ratio=1, random_state=RANDOM_SEED)

    matched = psm.match_by_group(
        data=df,
        treatment_col=TREATMENT_COL,
        score_cols=[SCORE_COL],
        groupby_col=GROUP_COL,
    )

    assert sum(matched[TREATMENT_COL] == 0) == sum(matched[TREATMENT_COL] != 0)


def test_nearest_neighbor_match_control_to_treatment(generate_unmatched_data):
    """
    Tests whether control to treatment matching is working. Does so
    by using:

        replace=True
        treatment_to_control=False
        ratio=2


    And testing if we get 2x the number of control matches than treatment
    """
    df, features = generate_unmatched_data()

    psm = NearestNeighborMatch(
        replace=True, ratio=2, treatment_to_control=False, random_state=RANDOM_SEED
    )
    matched = psm.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])
    assert 2 * sum(matched[TREATMENT_COL] == 0) == sum(matched[TREATMENT_COL] != 0)


def test_match_optimizer(generate_unmatched_data):
    df, features = generate_unmatched_data()

    optimizer = MatchOptimizer(
        treatment_col=TREATMENT_COL,
        ps_col=SCORE_COL,
        matching_covariates=[SCORE_COL],
        min_users_per_group=100,
        smd_cols=[SCORE_COL],
        dev_cols_transformations={SCORE_COL: np.mean},
    )

    matched = optimizer.search_best_match(df)

    assert sum(matched[TREATMENT_COL] == 0) == sum(matched[TREATMENT_COL] != 0)
