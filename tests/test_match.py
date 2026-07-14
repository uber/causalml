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

    ``match()`` computes the (from, to) index pairs internally but previously
    discarded them; only the joined dataframe was returned. This asserts the
    pairs are now exposed as the fitted attribute ``matched_indexes_`` for both
    the replacement and no-replacement paths, that the exposed indices are
    consistent with the returned matched dataframe, and that with
    ``replace=True, ratio=2`` the attribute captures the *pair* mapping (a
    single ``from`` index paired against multiple ``to`` indices) rather than
    the deduplicated from-set.
    """
    df, features = generate_unmatched_data()

    # Replacement path with ratio=2 -- exercises the NearestNeighbors branch.
    psm = NearestNeighborMatch(replace=True, ratio=2, random_state=RANDOM_SEED)
    matched = psm.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])

    assert hasattr(psm, "matched_indexes_")
    assert isinstance(psm.matched_indexes_, pd.DataFrame)
    assert set(psm.matched_indexes_.columns) == {"from", "to"}
    assert len(psm.matched_indexes_) > 0

    # Every from/to in the pair table must appear in the matched dataframe.
    matched_idx = set(matched.index)
    assert set(psm.matched_indexes_["from"].unique()).issubset(matched_idx)
    assert set(psm.matched_indexes_["to"].unique()).issubset(matched_idx)

    # ratio=2, replace=True: at least one `from` should pair with two `to`s.
    counts_per_from = psm.matched_indexes_["from"].value_counts()
    assert (counts_per_from >= 2).any()

    # No-replacement path (caliper loop) is also populated with the schema.
    psm2 = NearestNeighborMatch(replace=False, ratio=1, random_state=RANDOM_SEED)
    psm2.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])
    assert hasattr(psm2, "matched_indexes_")
    assert set(psm2.matched_indexes_.columns) == {"from", "to"}
    assert len(psm2.matched_indexes_) > 0


def test_nearest_neighbor_match_exhausts_control_pool():
    """Matching without replacement should not crash when the pool of
    unmatched controls shrinks below `ratio` during the loop.

    With a balanced 1:1 dataset the last treatment unit sees a single
    remaining control, which previously made np.argpartition raise
    "kth out of bounds". See match.py::NearestNeighborMatch.match.
    """
    df = pd.DataFrame(
        {
            SCORE_COL: [0.10, 0.40, 0.80, 0.12, 0.42, 0.83],
            TREATMENT_COL: [1, 1, 1, 0, 0, 0],
        }
    )

    psm = NearestNeighborMatch(
        replace=False, ratio=1, shuffle=False, random_state=RANDOM_SEED
    )
    matched = psm.match(data=df, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL])

    # All three balanced pairs are recovered.
    assert sum(matched[TREATMENT_COL] == 1) == 3
    assert sum(matched[TREATMENT_COL] == 0) == 3

    # More treatment than control: matching stops once controls run out
    # instead of raising.
    df_scarce = pd.DataFrame(
        {
            SCORE_COL: [0.10, 0.40, 0.80, 0.90, 0.12, 0.42],
            TREATMENT_COL: [1, 1, 1, 1, 0, 0],
        }
    )
    matched_scarce = psm.match(
        data=df_scarce, treatment_col=TREATMENT_COL, score_cols=[SCORE_COL]
    )
    assert sum(matched_scarce[TREATMENT_COL] == 0) == 2


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
