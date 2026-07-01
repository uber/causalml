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


def test_nearest_neighbor_match_exposes_matched_indexes(generate_unmatched_data):
    """Regression test for uber/causalml#621.

    The user requested access to the matched index pairs as an attribute.
    Previously the (from, to) pairs were computed inside ``match`` and
    discarded; only the joined dataframe was returned. This test asserts:

    1. ``matched_indexes_`` is set after ``match()`` runs.
    2. It has the documented schema (``from``, ``to``).
    3. The exposed pairs are consistent with the returned matched dataframe
       — every index in ``matched_indexes_['from']`` and ``['to']`` shows up
       in the matched dataframe's index.
    4. With ``replace=True, ratio=2``, a single ``from`` index can appear
       multiple times (one row per matched control), proving the attribute
       captures the *pair* mapping rather than the deduplicated set.
    """
    df, features = generate_unmatched_data()

    # Replacement path with ratio=2 — exercises the NearestNeighbors branch.
    psm = NearestNeighborMatch(replace=True, ratio=2, random_state=RANDOM_SEED)
    matched = psm.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])

    assert hasattr(
        psm, "matched_indexes_"
    ), "matched_indexes_ must be set after match() — see uber/causalml#621"
    assert isinstance(psm.matched_indexes_, pd.DataFrame)
    assert set(psm.matched_indexes_.columns) == {"from", "to"}
    assert len(psm.matched_indexes_) > 0

    # The matched dataframe is the union of unique from + all to indices, so
    # every from / to in the pair table must appear in the matched index.
    matched_idx = set(matched.index)
    assert set(psm.matched_indexes_["from"].unique()).issubset(matched_idx)
    assert set(psm.matched_indexes_["to"].unique()).issubset(matched_idx)

    # ratio=2 with replace=True: at least one `from` should have two `to`
    # matches, proving the attribute captures the pair mapping (not just
    # the deduplicated unique-from set).
    counts_per_from = psm.matched_indexes_["from"].value_counts()
    assert (counts_per_from >= 2).any(), (
        "With replace=True, ratio=2 the pair table must show at least one "
        "from-index paired against multiple to-indices"
    )

    # No-replacement path is also exercised by the existing tests; here we
    # just confirm the attribute is populated for that path too.
    psm2 = NearestNeighborMatch(replace=False, ratio=1, random_state=RANDOM_SEED)
    psm2.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])
    assert hasattr(psm2, "matched_indexes_")
    assert set(psm2.matched_indexes_.columns) == {"from", "to"}
    assert len(psm2.matched_indexes_) > 0


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
