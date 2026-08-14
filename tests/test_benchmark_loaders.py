"""Tests for the benchmark dataset loaders, against the real files.

Marked ``network`` and skipped unless the dataset is already cached, so the
ordinary suite never downloads anything. Run them with the cache warm, or on the
schedule that exists to catch the source URLs moving::

    pytest tests/test_benchmark_loaders.py -m network

Each assertion is a published property of the dataset rather than a snapshot of
whatever the file happens to contain, so a source that changes underneath us
fails here rather than propagating into a benchmark table.
"""

import numpy as np
import pytest

from causalml.dataset import fetch_ihdp, fetch_lalonde, fetch_twins
from causalml.metrics import ate_error, pehe

pytestmark = pytest.mark.network


def _cached(loader, **kwargs):
    """Load with downloads disabled, skipping the test when the cache is cold."""
    try:
        return loader(download_if_missing=False, **kwargs)
    except OSError as exc:
        pytest.skip(f"dataset not cached: {exc}")


def test_fetch_lalonde_reproduces_the_experimental_estimate():
    """445 rows, 185 treated, and the ~1794 dollar experimental difference."""
    lalonde = _cached(fetch_lalonde)

    assert lalonde.data.shape == (445, 8)
    assert lalonde.treatment.sum() == 185
    assert lalonde.feature_names[:2] == ["age", "educ"]

    treated = lalonde.target[lalonde.treatment == 1].mean()
    control = lalonde.target[lalonde.treatment == 0].mean()
    assert treated - control == pytest.approx(1794.34, abs=1.0)


def test_fetch_ihdp_replications_differ_and_carry_ground_truth():
    """672 x 25 per replication, with mu0/mu1 giving the individual effect.

    The replication axis is asserted because selecting the wrong one is silent:
    every replication is the same 747 units re-split and re-simulated, so any of
    them yields a plausible-looking dataset of the right shape.
    """
    first = _cached(fetch_ihdp, replication=0)
    seventh = _cached(fetch_ihdp, replication=7)

    assert first.data.shape == (672, 25)
    assert first.treatment.sum() == 123
    assert first.tau.mean() == pytest.approx(4.012, abs=0.01)
    assert np.allclose(first.tau, first.mu1 - first.mu0)

    # Each replication draws its own train/test split of the same 747 units, so
    # the rows are not aligned across replications -- only the pooled covariate
    # set is. Asserting row-wise equality would look reasonable and be false.
    assert not np.allclose(first.data, seventh.data)
    assert not np.allclose(first.tau, seventh.tau)
    assert first.treatment.sum() != seventh.treatment.sum()

    test_split = _cached(fetch_ihdp, split="test")
    assert test_split.data.shape == (75, 25)
    pooled_first = np.unique(np.vstack([first.data, test_split.data]), axis=0)
    pooled_seventh = np.unique(
        np.vstack(
            [seventh.data, _cached(fetch_ihdp, replication=7, split="test").data]
        ),
        axis=0,
    )
    assert pooled_first.shape == (747, 25)
    assert np.array_equal(pooled_first, pooled_seventh)


def test_fetch_ihdp_validates_its_arguments():
    """A replication outside the file, or an unknown split, says so."""
    with pytest.raises(ValueError, match="replication"):
        fetch_ihdp(replication=100, download_if_missing=False)
    with pytest.raises(ValueError, match="split"):
        fetch_ihdp(split="validation", download_if_missing=False)


def test_fetch_twins_mortality_matches_the_published_rate():
    """The 9999 sentinel decides whether this dataset means anything.

    Read as a number the outcome column averages about 8000; read as
    `outcome < 9999` it gives the 17.7% mortality for the lighter twin that
    Louizos et al. (2017) report. Asserting the rate pins the interpretation.
    """
    twins = _cached(fetch_twins, random_state=42)

    assert twins.data.shape == (11400, 30)
    assert twins.y0.mean() == pytest.approx(0.1769, abs=0.001)
    assert twins.y1.mean() == pytest.approx(0.1608, abs=0.001)
    assert set(np.unique(twins.y0)) <= {0, 1}


def test_fetch_twins_reveals_the_twin_the_assignment_picks():
    """The observed outcome is the assigned twin's, and `tau` keeps both."""
    twins = _cached(fetch_twins, random_state=42)

    assert np.array_equal(
        twins.target, np.where(twins.treatment == 1, twins.y1, twins.y0)
    )
    assert np.array_equal(twins.tau, twins.y1 - twins.y0)
    assert twins.tau.mean() == pytest.approx(-0.0161, abs=0.001)


def test_fetch_twins_assignment_is_seeded():
    """The same seed reveals the same twins; a different one does not."""
    a = _cached(fetch_twins, random_state=1)
    b = _cached(fetch_twins, random_state=1)
    c = _cached(fetch_twins, random_state=2)

    assert np.array_equal(a.treatment, b.treatment)
    assert not np.array_equal(a.treatment, c.treatment)


@pytest.mark.parametrize("loader", [fetch_lalonde, fetch_twins])
def test_return_X_y_t_matches_the_bunch(loader):
    """The triple is the same arrays the Bunch carries.

    Twins needs the seed: its assignment is random per call by design, so an
    unseeded pair of calls would differ for a reason that is not a bug.
    """
    seed = {} if loader is fetch_lalonde else {"random_state": 0}
    bunch = _cached(loader, **seed)
    X, y, treatment = _cached(loader, return_X_y_t=True, **seed)

    assert np.array_equal(X, bunch.data)
    assert np.array_equal(y, bunch.target)
    assert X.shape[0] == y.shape[0] == treatment.shape[0]


def test_ground_truth_metrics_run_on_the_datasets_that_have_it():
    """PEHE needs per-unit truth; the experiment only licenses an ATE error."""
    ihdp = _cached(fetch_ihdp, replication=0)

    assert pehe(ihdp.tau, ihdp.tau) == 0.0
    assert pehe(ihdp.tau, np.full_like(ihdp.tau, ihdp.tau.mean())) == pytest.approx(
        ihdp.tau.var()
    )

    lalonde = _cached(fetch_lalonde)
    experimental = (
        lalonde.target[lalonde.treatment == 1].mean()
        - lalonde.target[lalonde.treatment == 0].mean()
    )
    assert ate_error(experimental, experimental) == 0.0
