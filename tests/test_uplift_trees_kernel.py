"""Numerical parity tests for the kernel-backed uplift trees (epic #945, issue #946).

These assert that the experimental ``_KernelUpliftTreeClassifier`` -- which grows an
uplift tree on the shared ``_tree`` Cython kernel via the new
``UpliftClassificationCriterion`` -- reproduces the legacy ``UpliftTreeClassifier``
predictions exactly for the KL / ED / Chi / CTS / DDP / IT / CIT / IDDP criteria,
with or without regularization, normalization, and the honest approach (pruning is
handled in a later issue of the epic). A final section bags the kernel tree into
``_KernelUpliftRandomForestClassifier`` (issue #952) and checks the same
whole-forest parity against the legacy ``UpliftRandomForestClassifier``.

Design notes
------------
* **Binary features.** The kernel's exhaustive midpoint split search and the
  legacy tree's percentile-candidate search only evaluate the *same* set of
  partitions when features are low-cardinality. With binary features the two
  candidate sets coincide, so exact whole-tree parity is achievable; on
  continuous features they diverge (kernel considers strictly more thresholds)
  and only the per-node criterion math would match.
* **Depth convention.** The legacy builder counts depth from 1 and stops at
  ``depth < max_depth`` (so ``max_depth=D`` allows ``D-1`` splits deep), while the
  kernel counts from 0 and stops at ``depth >= max_depth``. Identical structure
  therefore needs ``legacy_max_depth = kernel_max_depth + 1``.
"""

import numpy as np
import pandas as pd
import pytest
from joblib import parallel_backend
from numpy.testing import assert_array_almost_equal
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import uplift_tree_plot, uplift_tree_string
from causalml.inference.tree.uplift import (
    UpliftTreeClassifier,
    UpliftRandomForestClassifier,
)
from causalml.inference.tree._uplift.uplifttree import _KernelUpliftTreeClassifier
from causalml.inference.tree._uplift.upliftforest import (
    _KernelUpliftRandomForestClassifier,
)
from causalml.metrics import auuc_score

from .const import RANDOM_SEED, CONTROL_NAME, TREATMENT_NAMES, CONVERSION

KERNEL_UPLIFT_CRITERIA = ["KL", "ED", "Chi"]

# legacy_max_depth = kernel_max_depth + LEGACY_DEPTH_OFFSET (see module docstring).
LEGACY_DEPTH_OFFSET = 1


def _make_binary_feature_data(
    n_samples=3000, n_features=5, treatment_names=("treatment1",), seed=RANDOM_SEED
):
    """Generate uplift data with binary features and heterogeneous, group-specific lifts."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_features)).astype(np.float32)
    groups = [CONTROL_NAME, *treatment_names]
    treatment = np.array(groups)[rng.randint(0, len(groups), size=n_samples)]

    base = 0.3 + 0.1 * X[:, 1] - 0.05 * X[:, 2]
    p = base.copy()
    for i, tr in enumerate(treatment_names):
        feat = min(3 + i, n_features - 1)
        lift = 0.25 * X[:, 0] + 0.1 * X[:, feat]
        p = np.where(treatment == tr, base + lift, p)
    y = (rng.rand(n_samples) < np.clip(p, 0.0, 1.0)).astype(int)
    return X, treatment, y


def _make_normalization_sensitive_data(n_samples=2000, n_features=6, seed=1):
    """Binary-feature uplift data on which Rzepakowski normalization re-ranks splits.

    Unlike :func:`_make_binary_feature_data` (balanced 50/50 assignment, so every
    candidate split has a near-constant normalization factor and normalization is
    effectively a no-op), here treatment assignment is *correlated with* ``X[:, 0]``.
    The treatment/control balance -- and hence the per-split normalization factor
    ``arr_normI`` -- then varies across candidate splits, so ``normalization=True``
    genuinely changes the chosen splits. This is what exercises ``_norm_factor``
    (and would catch it reading the wrong child).
    """
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_features)).astype(np.float32)
    ptreat = np.where(X[:, 0] > 0, 0.75, 0.30)
    treatment = np.where(rng.rand(n_samples) < ptreat, "treatment1", CONTROL_NAME)
    is_t = treatment == "treatment1"
    base = 0.35 + 0.08 * X[:, 1] - 0.04 * X[:, 2] + 0.05 * X[:, 3]
    lift = 0.12 * X[:, 0] + 0.08 * X[:, 4] - 0.05 * X[:, 5]
    p = np.where(is_t, base + lift, base)
    y = (rng.rand(n_samples) < np.clip(p, 0.0, 1.0)).astype(int)
    return X, treatment, y


def _fit_pair(
    X,
    treatment,
    y,
    criterion,
    kernel_max_depth,
    min_samples_leaf=100,
    n_reg=0,
    min_samples_treatment=0,
    normalization=False,
):
    """Fit the kernel-backed tree and the parity-configured legacy tree.

    ``n_reg`` / ``min_samples_treatment`` (issue #947) and ``normalization``
    (issue #948) are passed identically to both trees so the regularization and
    normalization can be exercised; the defaults disable them for the plain parity
    cases.
    """
    kern = _KernelUpliftTreeClassifier(
        criterion=criterion,
        control_name=CONTROL_NAME,
        max_depth=kernel_max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        n_reg=n_reg,
        normalization=normalization,
        random_state=RANDOM_SEED,
    )
    kern.fit(X, treatment, y)

    legacy = UpliftTreeClassifier(
        control_name=CONTROL_NAME,
        evaluationFunction=criterion,
        max_depth=kernel_max_depth + LEGACY_DEPTH_OFFSET,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        n_reg=n_reg,
        normalization=normalization,
        honesty=False,
        random_state=RANDOM_SEED,
    )
    legacy.fit(X, treatment, y)
    return kern, legacy


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_single_treatment(criterion, kernel_max_depth):
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair(X, treatment, y, criterion, kernel_max_depth)

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    # Same leaf P(Y=1|T=g) for every group, hence identical trees.
    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
def test_kernel_uplift_parity_multi_treatment(criterion):
    X, treatment, y = _make_binary_feature_data(
        n_samples=4500,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=7,
    )
    kern, legacy = _fit_pair(X, treatment, y, criterion, kernel_max_depth=2)

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert kernel_proba.shape[1] == 3  # control + 2 treatments
    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
def test_kernel_uplift_predict_deltas(criterion):
    """`predict` returns per-treatment deltas P(Y=1|T=t) - P(Y=1|control)."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair(X, treatment, y, criterion, kernel_max_depth=2)

    proba = kern.predict_proba_by_group(X)
    deltas = kern.predict(X)

    assert deltas.shape == (X.shape[0], proba.shape[1] - 1)
    assert_array_almost_equal(deltas, proba[:, 1:] - proba[:, [0]], decimal=12)
    # deltas match legacy-derived deltas too.
    legacy_proba = legacy.predict(X)
    assert_array_almost_equal(
        deltas, legacy_proba[:, 1:] - legacy_proba[:, [0]], decimal=8
    )


def test_kernel_uplift_zero_divergence_root_still_splits():
    """A near-zero-ATE root must still split (regression guard).

    The stock kernel builder turns a node into a leaf when its impurity is
    ~0; uplift growth is instead driven by the split gain, so a root with no
    overall treatment/control divergence must still be splittable when a split
    creates divergent children.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n = 4000
    X = rng.randint(0, 2, size=(n, 3)).astype(np.float32)
    treatment = np.where(rng.rand(n) < 0.5, CONTROL_NAME, "treatment1")
    is_t = treatment == "treatment1"
    # Effect flips sign with X[:,0] => overall ATE ~ 0 at the root, strong
    # heterogeneity below it.
    p = 0.4 + np.where(X[:, 0] > 0, 0.2, -0.2) * is_t
    y = (rng.rand(n) < np.clip(p, 0.0, 1.0)).astype(int)

    kern = _KernelUpliftTreeClassifier(
        criterion="ED",
        control_name=CONTROL_NAME,
        max_depth=2,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    kern.fit(X, treatment, y)
    assert kern.tree_.node_count > 1  # root actually split


# --- Regularization parity (issue #947) -------------------------------------
# Legacy public defaults: n_reg=100, min_samples_treatment=10.
LEGACY_N_REG = 100
LEGACY_MIN_SAMPLES_TREATMENT = 10


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_regularized(criterion, kernel_max_depth):
    """Whole-tree parity with Rzepakowski parent-shrinkage regularization on."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
def test_kernel_uplift_parity_regularized_multi_treatment(criterion):
    X, treatment, y = _make_binary_feature_data(
        n_samples=4500,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=7,
    )
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth=2,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert kernel_proba.shape[1] == 3  # control + 2 treatments
    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


def test_kernel_uplift_min_samples_treatment_gates_splits():
    """``min_samples_treatment`` rejects splits whose child groups are too small.

    With the floor set above any achievable per-group child size, no candidate
    split is admissible and the tree collapses to its root; dropping the floor to
    zero (same data / depth / leaf size) lets it grow -- proving the gate, not some
    other stopping rule, is what prevented growth.
    """
    X, treatment, y = _make_binary_feature_data()

    gated = _KernelUpliftTreeClassifier(
        criterion="ED",
        control_name=CONTROL_NAME,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_treatment=X.shape[0],
        n_reg=0,
        random_state=RANDOM_SEED,
    )
    gated.fit(X, treatment, y)
    assert gated.tree_.node_count == 1

    ungated = _KernelUpliftTreeClassifier(
        criterion="ED",
        control_name=CONTROL_NAME,
        max_depth=3,
        min_samples_leaf=1,
        min_samples_treatment=0,
        n_reg=0,
        random_state=RANDOM_SEED,
    )
    ungated.fit(X, treatment, y)
    assert ungated.tree_.node_count > 1


# --- Normalization + CTS parity (issue #948) --------------------------------


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_normalized(criterion, kernel_max_depth):
    """Whole-tree parity with Rzepakowski normalization on (KL/ED/Chi)."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
def test_kernel_uplift_parity_normalized_multi_treatment(criterion):
    X, treatment, y = _make_binary_feature_data(
        n_samples=4500,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=7,
    )
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth=2,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert kernel_proba.shape[1] == 3  # control + 2 treatments
    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


@pytest.mark.parametrize("criterion", KERNEL_UPLIFT_CRITERIA)
def test_kernel_uplift_parity_normalization_effective(criterion):
    """Normalized parity on data where normalization actually re-ranks splits.

    The balanced-assignment parity tests above run on data where normalization
    never changes the chosen split, so they cannot detect a wrong
    ``_norm_factor`` (e.g. one built from the wrong child). Here treatment is
    correlated with ``X[:, 0]`` so normalization flips the tree; the kernel must
    (a) still match the legacy tree exactly *and* (b) differ from the
    unnormalized tree -- otherwise the flag is a silent no-op.
    """
    X, treatment, y = _make_normalization_sensitive_data()
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth=3,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )
    plain, _ = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth=3,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=False,
    )
    kern_proba = kern.predict_proba_by_group(X)

    # (a) exact parity with the legacy normalized tree.
    assert_array_almost_equal(kern_proba, legacy.predict(X), decimal=8)
    # (b) normalization is not silently ignored: it changes the fitted tree here.
    assert not np.allclose(kern_proba, plain.predict_proba_by_group(X))


@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_cts_single_treatment(kernel_max_depth):
    """CTS parity with legacy defaults (n_reg=100, mst=10). CTS ignores normalization."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        "CTS",
        kernel_max_depth,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


def test_kernel_uplift_parity_cts_multi_treatment():
    X, treatment, y = _make_binary_feature_data(
        n_samples=4500,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=7,
    )
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        "CTS",
        kernel_max_depth=2,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert kernel_proba.shape[1] == 3  # control + 2 treatments
    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


# --- Two-class / variance criteria: DDP, IT, CIT (issue #949) ----------------
# These contrast a single treatment against control (two-class only) and are
# never normalized. (IDDP is deferred to the honesty issue #950, since legacy
# forces honesty=True for IDDP and the kernel has no honesty pass yet.)
TWO_CLASS_CRITERIA = ["DDP", "IT", "CIT"]


@pytest.mark.parametrize("criterion", TWO_CLASS_CRITERIA)
@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_two_class(criterion, kernel_max_depth):
    """DDP/IT/CIT parity with legacy defaults (n_reg=100, mst=10), single treatment."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,  # ignored by these criteria; matches legacy defaults
    )

    kernel_proba = kern.predict_proba_by_group(X)
    legacy_proba = legacy.predict(X)

    assert_array_almost_equal(kernel_proba, legacy_proba, decimal=8)


@pytest.mark.parametrize("criterion", TWO_CLASS_CRITERIA)
def test_kernel_uplift_two_class_guard_rejects_multi_treatment(criterion):
    """DDP/IT/CIT must reject more than one treatment group (legacy uplift.pyx)."""
    X, treatment, y = _make_binary_feature_data(
        n_samples=3000,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=7,
    )
    kern = _KernelUpliftTreeClassifier(
        criterion=criterion,
        control_name=CONTROL_NAME,
        max_depth=2,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    with pytest.raises(ValueError, match="two-class"):
        kern.fit(X, treatment, y)


# --- Honest approach + IDDP (issue #950) -------------------------------------
# Honesty grows the tree on a training split and re-estimates each leaf's
# per-group P(Y=1|T=g) on a held-out estimation split (Athey & Imbens 2016). The
# kernel mirrors the legacy split exactly -- same stratify=(treatment, y),
# test_size, shuffle, random_state -- so the two trees partition the data
# identically and the re-estimated leaves match to machine precision. IDDP is
# grown here because legacy forces honesty=True for it.
HONEST_CRITERIA = ["KL", "ED", "Chi", "CTS", "DDP", "IT", "CIT"]


def _fit_pair_honest(
    X,
    treatment,
    y,
    criterion,
    kernel_max_depth,
    min_samples_leaf=100,
    n_reg=0,
    min_samples_treatment=0,
    normalization=False,
    estimation_sample_size=0.5,
):
    """Fit kernel + legacy trees with the honest approach enabled on both.

    Both take the same ``random_state`` / ``estimation_sample_size`` so their
    train/estimation splits -- and hence their re-estimated leaves -- coincide.
    """
    kern = _KernelUpliftTreeClassifier(
        criterion=criterion,
        control_name=CONTROL_NAME,
        max_depth=kernel_max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        n_reg=n_reg,
        normalization=normalization,
        honesty=True,
        estimation_sample_size=estimation_sample_size,
        random_state=RANDOM_SEED,
    )
    kern.fit(X, treatment, y)

    legacy = UpliftTreeClassifier(
        control_name=CONTROL_NAME,
        evaluationFunction=criterion,
        max_depth=kernel_max_depth + LEGACY_DEPTH_OFFSET,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        n_reg=n_reg,
        normalization=normalization,
        honesty=True,
        estimation_sample_size=estimation_sample_size,
        random_state=RANDOM_SEED,
    )
    legacy.fit(X, treatment, y)
    return kern, legacy


def _make_iddp_normalization_sensitive_data(n_samples=3000, n_features=5, seed=11):
    """Binary-feature two-class data on which IDDP normalization re-ranks splits.

    IDDP is only ever grown honestly, and its normalization branch (a distinct
    ``arr_normI`` variant whose factor depends on the raw gain) is a no-op on the
    balanced-assignment data. As with :func:`_make_normalization_sensitive_data`,
    correlating treatment with ``X[:, 0]`` makes the per-split factor vary so
    ``normalization=True`` genuinely flips the IDDP tree -- the only setup that can
    catch ``_norm_factor`` reading the wrong (asymmetric) child.
    """
    rng = np.random.RandomState(seed)
    X = (rng.rand(n_samples, n_features) > 0.5).astype(np.float32)
    ptreat = np.where(X[:, 0] > 0, 0.9, 0.1)
    treat = rng.rand(n_samples) < ptreat
    treatment = np.where(treat, "treatment1", CONTROL_NAME)
    base = 0.2 + 0.3 * X[:, 1]
    lift = np.where(X[:, 2] > 0, 0.3, -0.1) * treat
    y = (rng.rand(n_samples) < np.clip(base + lift, 0.0, 1.0)).astype(int)
    return X, treatment, y


@pytest.mark.parametrize("criterion", HONEST_CRITERIA)
@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_honest(criterion, kernel_max_depth):
    """Honest-tree parity with legacy defaults (n_reg=100, mst=10), single treatment."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair_honest(
        X,
        treatment,
        y,
        criterion,
        kernel_max_depth,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,  # ignored by CTS/DDP/IT/CIT; matches legacy defaults
    )

    assert_array_almost_equal(
        kern.predict_proba_by_group(X), legacy.predict(X), decimal=8
    )


def test_kernel_uplift_honesty_changes_leaves():
    """Honesty is not a silent no-op: re-estimated leaves differ from the plain tree.

    The honest tree grows on the training half and re-estimates leaves on the
    held-out half, so its leaf probabilities must differ from a non-honest tree
    fit on the full data.
    """
    X, treatment, y = _make_binary_feature_data()
    honest, _ = _fit_pair_honest(
        X,
        treatment,
        y,
        "KL",
        kernel_max_depth=3,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
    )
    plain, _ = _fit_pair(
        X,
        treatment,
        y,
        "KL",
        kernel_max_depth=3,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
    )
    assert not np.allclose(
        honest.predict_proba_by_group(X), plain.predict_proba_by_group(X)
    )


@pytest.mark.parametrize("kernel_max_depth", [1, 2, 3])
def test_kernel_uplift_parity_iddp(kernel_max_depth):
    """IDDP parity (honesty forced) with legacy defaults, single treatment."""
    X, treatment, y = _make_binary_feature_data()
    kern, legacy = _fit_pair_honest(
        X,
        treatment,
        y,
        "IDDP",
        kernel_max_depth,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )

    assert_array_almost_equal(
        kern.predict_proba_by_group(X), legacy.predict(X), decimal=8
    )


def test_kernel_uplift_parity_iddp_normalization_effective():
    """IDDP normalized parity on data where its ``arr_normI`` branch re-ranks splits.

    IDDP normalization is a no-op on balanced-assignment data, so this uses
    treatment correlated with ``X[:, 0]`` (like the KL/ED/Chi normalization test)
    to force the tree to flip. The kernel must (a) still match the legacy honest
    tree exactly *and* (b) differ from the unnormalized IDDP tree -- which is what
    exercises the gain-dependent ``_norm_factor`` and its asymmetric child.
    """
    X, treatment, y = _make_iddp_normalization_sensitive_data()
    kern, legacy = _fit_pair_honest(
        X,
        treatment,
        y,
        "IDDP",
        kernel_max_depth=3,
        min_samples_leaf=50,
        n_reg=LEGACY_N_REG,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        normalization=True,
    )
    plain = _KernelUpliftTreeClassifier(
        criterion="IDDP",
        control_name=CONTROL_NAME,
        max_depth=3,
        min_samples_leaf=50,
        min_samples_treatment=LEGACY_MIN_SAMPLES_TREATMENT,
        n_reg=LEGACY_N_REG,
        normalization=False,
        random_state=RANDOM_SEED,
    )
    plain.fit(X, treatment, y)

    kern_proba = kern.predict_proba_by_group(X)
    # (a) exact parity with the legacy normalized honest tree.
    assert_array_almost_equal(kern_proba, legacy.predict(X), decimal=8)
    # (b) normalization is not silently ignored: it changes the fitted IDDP tree.
    assert not np.allclose(kern_proba, plain.predict_proba_by_group(X))


def test_kernel_uplift_iddp_forces_honesty():
    """IDDP forces the honest approach at fit, without mutating stored honesty."""
    X, treatment, y = _make_binary_feature_data()
    common = dict(
        criterion="IDDP",
        control_name=CONTROL_NAME,
        max_depth=3,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    forced = _KernelUpliftTreeClassifier(honesty=False, **common).fit(X, treatment, y)
    explicit = _KernelUpliftTreeClassifier(honesty=True, **common).fit(X, treatment, y)
    # honesty=False is forced onto the honest path in fit, matching honesty=True,
    # while the constructor keeps the argument verbatim (sklearn clone/get_params).
    assert forced.get_params()["honesty"] is False
    assert_array_almost_equal(
        forced.predict_proba_by_group(X),
        explicit.predict_proba_by_group(X),
        decimal=12,
    )


def test_kernel_uplift_iddp_guard_rejects_multi_treatment():
    """IDDP is two-class only and must reject more than one treatment group."""
    X, treatment, y = _make_binary_feature_data(
        n_samples=3000,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=7,
    )
    kern = _KernelUpliftTreeClassifier(
        criterion="IDDP",
        control_name=CONTROL_NAME,
        max_depth=2,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    with pytest.raises(ValueError, match="two-class"):
        kern.fit(X, treatment, y)


# ---------------------------------------------------------------------------
# Plot compatibility (issue #953): the kernel tree exposes a legacy-shaped
# `fitted_uplift_tree`, so plot.py's helpers render it unchanged.
# ---------------------------------------------------------------------------


def _assert_plot_node_tree(root, classes_, node_count):
    """Walk the adapter tree, checking the fields plot.py duck-types."""

    def walk(node):
        if node.results is not None:  # leaf
            assert node.col == -1 and node.value is None
            assert node.trueBranch is None and node.falseBranch is None
            assert len(node.results) == len(classes_)
            return 1
        assert node.results is None
        assert isinstance(node.col, int) and isinstance(node.value, float)
        assert node.trueBranch is not None and node.falseBranch is not None
        return 1 + walk(node.trueBranch) + walk(node.falseBranch)

    assert walk(root) == node_count
    for node in (root,):
        for key in ("impurity", "samples", "group_size", "upliftScore", "matchScore"):
            assert key in node.summary
        assert len(node.summary["upliftScore"]) == 2


@pytest.mark.parametrize("honesty", [False, True])
def test_kernel_uplift_tree_visualization(honesty):
    """`uplift_tree_plot(...).create_png()` renders the kernel tree (mirrors the
    legacy `test_uplift_tree_visualization`)."""
    df, x_names = make_uplift_classification(random_seed=RANDOM_SEED)
    df = df[df["treatment_group_key"].isin([CONTROL_NAME, "treatment1"])]

    model = _KernelUpliftTreeClassifier(
        criterion="KL",
        control_name=CONTROL_NAME,
        max_depth=4,
        min_samples_leaf=200,
        min_samples_treatment=50,
        n_reg=100,
        honesty=honesty,
        random_state=RANDOM_SEED,
    )
    model.fit(
        df[x_names].values,
        df["treatment_group_key"].values,
        df["conversion"].values,
    )
    root = model.fitted_uplift_tree

    # Renders to a PNG without raising (the legacy smoke assertion).
    png = uplift_tree_plot(root, x_names).create_png()
    assert len(png) > 0
    uplift_tree_string(root, x_names)  # text renderer smoke

    _assert_plot_node_tree(root, model.classes_, model.tree_.node_count)
    # group_size lists every group; root sample count is the fit-split size.
    assert root.summary["group_size"].count(":") == len(model.classes_)
    assert int(root.summary["samples"]) == model.tree_.n_node_samples[0]


def test_kernel_uplift_tree_plot_multi_treatment():
    """The adapter and plot handle more than one treatment group."""
    df, x_names = make_uplift_classification(random_seed=RANDOM_SEED)  # 4 groups

    model = _KernelUpliftTreeClassifier(
        criterion="KL",
        control_name=CONTROL_NAME,
        max_depth=3,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    model.fit(
        df[x_names].values,
        df["treatment_group_key"].values,
        df["conversion"].values,
    )
    root = model.fitted_uplift_tree

    assert len(model.classes_) == 4
    _assert_plot_node_tree(root, model.classes_, model.tree_.node_count)
    assert len(uplift_tree_plot(root, x_names).create_png()) > 0


def test_kernel_uplift_tree_fitted_tree_requires_fit():
    """`fitted_uplift_tree` on an unfitted estimator raises, not returns junk."""
    model = _KernelUpliftTreeClassifier(control_name=CONTROL_NAME)
    with pytest.raises(Exception):
        _ = model.fitted_uplift_tree


# --- Validation-based pruning (issue #951) -----------------------------------
# The kernel ``prune`` collapses an internal node whose two children are leaves
# into a leaf when the split does not improve the treatment effect on a held-out
# validation set, cascading upward, then rebuilds a compact tree. Legacy
# ``UpliftTreeClassifier.prune`` is a no-op (its recursion never descends past
# the root), so these are behavioral checks, not legacy bit-parity.
PRUNE_RULES = ["maxAbsDiff", "bestUplift"]


def _make_prune_signal_data(n_samples=4000, n_features=6, seed=RANDOM_SEED):
    """Feature 0 drives a strong, generalizing uplift sign flip; the rest noise.

    A deep tree over-splits on the noise features; only the feature-0 split
    reproduces on an independent validation draw, so pruning should drop the
    noise splits and keep the feature-0 split.
    """
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_features)).astype(np.float32)
    treatment = np.where(rng.rand(n_samples) < 0.5, "treatment1", CONTROL_NAME)
    lift = np.where(X[:, 0] > 0, 0.30, -0.30)
    p = np.where(treatment == "treatment1", 0.4 + lift, 0.4)
    y = (rng.rand(n_samples) < np.clip(p, 0.0, 1.0)).astype(int)
    return X, treatment, y


def _fit_prunable_tree(X, treatment, y, max_depth=4):
    """A deep, unregularized tree that over-splits (so there is something to prune)."""
    kern = _KernelUpliftTreeClassifier(
        criterion="KL",
        control_name=CONTROL_NAME,
        max_depth=max_depth,
        min_samples_leaf=50,
        n_reg=0,
        min_samples_treatment=0,
        normalization=False,
        random_state=RANDOM_SEED,
    )
    kern.fit(X, treatment, y)
    return kern


def _reachable_node_ids(tree):
    """Node ids reachable from the root by following children_left/right."""
    left, right = tree.children_left, tree.children_right
    seen, stack = set(), [0]
    while stack:
        node = stack.pop()
        if node == -1 or node in seen:
            continue
        seen.add(node)
        stack.extend((left[node], right[node]))
    return seen


@pytest.mark.parametrize("rule", PRUNE_RULES)
def test_kernel_uplift_prune_reduces_and_stays_compact(rule):
    X, t, y = _make_prune_signal_data(seed=RANDOM_SEED)
    Xv, tv, yv = _make_prune_signal_data(seed=RANDOM_SEED + 1)
    kern = _fit_prunable_tree(X, t, y)
    before = kern.tree_.node_count
    kern.prune(Xv, tv, yv, minGain=0.005, rule=rule)
    after = kern.tree_.node_count

    assert after < before  # pruning removed nodes
    # Compact rebuild: every id < node_count is reachable, no orphans.
    assert _reachable_node_ids(kern.tree_) == set(range(after))
    left, right = kern.tree_.children_left, kern.tree_.children_right
    assert ((left == -1) == (right == -1)).all()  # leaf iff no children


@pytest.mark.parametrize("rule", PRUNE_RULES)
def test_kernel_uplift_prune_idempotent(rule):
    X, t, y = _make_prune_signal_data(seed=RANDOM_SEED)
    Xv, tv, yv = _make_prune_signal_data(seed=RANDOM_SEED + 1)
    kern = _fit_prunable_tree(X, t, y)
    kern.prune(Xv, tv, yv, minGain=0.005, rule=rule)
    once = kern.tree_.node_count
    proba_once = kern.predict_proba_by_group(X)

    kern.prune(Xv, tv, yv, minGain=0.005, rule=rule)
    assert kern.tree_.node_count == once
    assert_array_almost_equal(kern.predict_proba_by_group(X), proba_once, decimal=12)


@pytest.mark.parametrize("rule", PRUNE_RULES)
def test_kernel_uplift_prune_mingain_monotonic(rule):
    X, t, y = _make_prune_signal_data(seed=RANDOM_SEED)
    Xv, tv, yv = _make_prune_signal_data(seed=RANDOM_SEED + 1)
    counts = []
    for min_gain in (0.0, 0.001, 0.01, 0.05, 0.2):
        kern = _fit_prunable_tree(X, t, y)
        kern.prune(Xv, tv, yv, minGain=min_gain, rule=rule)
        counts.append(kern.tree_.node_count)
    # Higher minGain prunes at least as aggressively.
    assert all(a >= b for a, b in zip(counts, counts[1:]))


@pytest.mark.parametrize("rule", PRUNE_RULES)
def test_kernel_uplift_prune_keeps_generalizing_split(rule):
    """The one generalizing split survives aggressive pruning; noise is dropped."""
    X, t, y = _make_prune_signal_data(seed=RANDOM_SEED)
    Xv, tv, yv = _make_prune_signal_data(seed=RANDOM_SEED + 1)
    kern = _fit_prunable_tree(X, t, y)
    kern.prune(Xv, tv, yv, minGain=0.05, rule=rule)

    # Collapses to the single feature-0 split: root + two leaves.
    assert kern.tree_.node_count == 3
    assert kern.tree_.feature[0] == 0
    # Predictions still recover the feature-0 uplift sign flip.
    uplift = kern.predict(X)[:, 0]
    assert uplift[X[:, 0] == 1].mean() > 0.1
    assert uplift[X[:, 0] == 0].mean() < -0.1


@pytest.mark.parametrize("rule", PRUNE_RULES)
def test_kernel_uplift_prune_predictions_valid(rule):
    X, t, y = _make_prune_signal_data(seed=RANDOM_SEED)
    Xv, tv, yv = _make_prune_signal_data(seed=RANDOM_SEED + 1)
    kern = _fit_prunable_tree(X, t, y)
    kern.prune(Xv, tv, yv, minGain=0.01, rule=rule)

    proba = kern.predict_proba_by_group(X)
    assert proba.shape == (X.shape[0], kern.n_outputs_)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()


def test_kernel_uplift_prune_invalid_rule_raises():
    X, t, y = _make_prune_signal_data(seed=RANDOM_SEED)
    kern = _fit_prunable_tree(X, t, y)
    with pytest.raises(ValueError, match="maxAbsDiff"):
        kern.prune(X, t, y, rule="bogus")


# ---------------------------------------------------------------------------
# Forest (issue #952): _KernelUpliftRandomForestClassifier bags the kernel tree
# on sklearn's ForestRegressor scaffolding, replacing the legacy joblib forest.
# ---------------------------------------------------------------------------

FOREST_CRITERIA = ["KL", "ED", "Chi"]


def _fit_forest_pair(
    X,
    treatment,
    y,
    criterion,
    kernel_max_depth,
    n_estimators=5,
    normalization=False,
    honesty=False,
    random_state=RANDOM_SEED,
):
    """Fit the kernel forest and a structurally-identical legacy forest.

    Both forests draw per-tree seeds from the same parent RNG and bootstrap the
    same rows, so with binary features (identical split candidates), all features
    considered, and ``legacy_max_depth = kernel_max_depth + 1`` every tree -- and
    therefore the averaged forest -- matches to full precision.
    """
    n_features = X.shape[1]
    kern = _KernelUpliftRandomForestClassifier(
        control_name=CONTROL_NAME,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=kernel_max_depth,
        min_samples_leaf=100,
        min_samples_treatment=0,
        n_reg=0,
        normalization=normalization,
        honesty=honesty,
        max_features=None,
        random_state=random_state,
    )
    kern.fit(X, treatment, y)

    legacy = UpliftRandomForestClassifier(
        control_name=CONTROL_NAME,
        n_estimators=n_estimators,
        evaluationFunction=criterion,
        max_depth=kernel_max_depth + LEGACY_DEPTH_OFFSET,
        min_samples_leaf=100,
        min_samples_treatment=0,
        n_reg=0,
        normalization=normalization,
        honesty=honesty,
        max_features=n_features,
        random_state=random_state,
    )
    legacy.fit(X, treatment, y)
    return kern, legacy


@pytest.mark.parametrize("criterion", FOREST_CRITERIA)
@pytest.mark.parametrize("normalization", [False, True])
def test_kernel_uplift_forest_parity(criterion, normalization):
    """The kernel forest reproduces the legacy forest's uplift deltas exactly."""
    X, treatment, y = _make_binary_feature_data(
        n_samples=3000,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=RANDOM_SEED,
    )
    kern, legacy = _fit_forest_pair(
        X, treatment, y, criterion, kernel_max_depth=3, normalization=normalization
    )
    assert kern.classes_ == legacy.classes_
    assert_array_almost_equal(kern.predict(X), legacy.predict(X), decimal=8)


def test_kernel_uplift_forest_honest_parity():
    """Honest re-estimation carries through to whole-forest parity."""
    X, treatment, y = _make_binary_feature_data(
        n_samples=4000,
        n_features=6,
        treatment_names=("treatment1",),
        seed=RANDOM_SEED,
    )
    kern, legacy = _fit_forest_pair(
        X, treatment, y, "KL", kernel_max_depth=2, honesty=True
    )
    assert_array_almost_equal(kern.predict(X), legacy.predict(X), decimal=8)


def test_kernel_uplift_forest_predict_shape_with_sparse_groups():
    """#569: row-bootstrap can miss whole treatment groups; predict stays well-shaped.

    Minority groups with a single sample are excluded from most bootstraps, so
    some trees see a strict subset of the forest's classes (and some collapse to
    control-only and are dropped). ``_align_tree_predict`` must still yield a
    full-width, NaN-free prediction, identically in serial and parallel.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n = 102
    X = rng.randn(n, 3).astype(np.float32)
    treatment = np.array(
        [CONTROL_NAME] * 100 + [TREATMENT_NAMES[1]] + [TREATMENT_NAMES[2]]
    )
    y = rng.randint(0, 2, n)

    model = _KernelUpliftRandomForestClassifier(
        control_name=CONTROL_NAME,
        n_estimators=10,
        min_samples_leaf=1,
        min_samples_treatment=0,
        max_features=None,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )
    model.fit(X, treatment, y)

    # At least one surviving tree missed a group (the condition _align handles).
    assert any(len(t.classes_) < len(model.classes_) for t in model.estimators_)

    preds = model.predict(X)
    assert preds.shape == (n, len(model.classes_) - 1)
    assert not np.any(np.isnan(preds))

    with parallel_backend("loky", n_jobs=2):
        preds_par = model.predict(X)
    assert np.allclose(preds, preds_par)


@pytest.mark.parametrize("backend", ["loky", "threading", "multiprocessing"])
@pytest.mark.parametrize("joblib_prefer", ["threads", "processes"])
def test_kernel_uplift_forest_backend_determinism_and_auuc(
    generate_classification_data, backend, joblib_prefer
):
    """Predictions are backend-invariant and beat random on AUUC.

    Mirrors the legacy 12-case forest test (backends x joblib_prefer); the
    early_stopping axis is dropped because the kernel tree has no validation-set
    early stopping.
    """
    df, x_names = generate_classification_data()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    with parallel_backend(backend):
        model = _KernelUpliftRandomForestClassifier(
            control_name=CONTROL_NAME,
            min_samples_leaf=50,
            random_state=RANDOM_SEED,
            joblib_prefer=joblib_prefer,
        )
        model.fit(
            df_train[x_names].values,
            df_train["treatment_group_key"].values,
            df_train[CONVERSION].values,
        )

        predictions = {"single": model.predict(df_test[x_names].values)}
        for name, be in [
            ("loky", "loky"),
            ("threading", "threading"),
            ("mp", "multiprocessing"),
        ]:
            with parallel_backend(be, n_jobs=2):
                predictions[name] = model.predict(df_test[x_names].values)

    values = list(predictions.values())
    assert all(np.array_equal(values[0], rest) for rest in values[1:])

    result = pd.DataFrame(values[0], columns=model.classes_[1:])
    best_treatment = np.where(
        (result < 0).all(axis=1), CONTROL_NAME, result.idxmax(axis=1)
    )
    actual_is_best = np.where(df_test["treatment_group_key"] == best_treatment, 1, 0)
    actual_is_control = np.where(df_test["treatment_group_key"] == CONTROL_NAME, 1, 0)
    synthetic = (actual_is_best == 1) | (actual_is_control == 1)
    synth = result[synthetic]
    auuc_metrics = synth.assign(
        is_treated=1 - actual_is_control[synthetic],
        conversion=df_test.loc[synthetic, CONVERSION].values,
        treatment_effect=df_test.loc[synthetic, "treatment_effect"].values,
        uplift_tree=synth.max(axis=1),
    ).drop(columns=list(model.classes_[1:]))
    auuc = auuc_score(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="is_treated",
        treatment_effect_col="treatment_effect",
        normalize=True,
    )
    assert auuc["uplift_tree"] > 0.5


def test_kernel_uplift_forest_full_output():
    """full_output returns the per-group / recommendation / delta frame."""
    X, treatment, y = _make_binary_feature_data(
        n_samples=2000,
        n_features=6,
        treatment_names=("treatment1", "treatment2"),
        seed=RANDOM_SEED,
    )
    model = _KernelUpliftRandomForestClassifier(
        control_name=CONTROL_NAME,
        n_estimators=5,
        max_depth=3,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    model.fit(X, treatment, y)

    delta = model.predict(X)
    full = model.predict(X, full_output=True)
    expected_cols = (
        list(model.classes_)
        + ["recommended_treatment"]
        + [f"delta_{g}" for g in model.classes_[1:]]
        + ["max_delta"]
    )
    assert list(full.columns) == expected_cols
    assert full.shape[0] == X.shape[0]
    # The bare-array predict is exactly the delta columns of the full frame.
    assert_array_almost_equal(
        delta, full[[f"delta_{g}" for g in model.classes_[1:]]].values
    )


def test_kernel_uplift_forest_feature_importances():
    """feature_importances_ averages the trees and is a normalized distribution."""
    X, treatment, y = _make_binary_feature_data(
        n_samples=2000,
        n_features=6,
        treatment_names=("treatment1",),
        seed=RANDOM_SEED,
    )
    model = _KernelUpliftRandomForestClassifier(
        control_name=CONTROL_NAME,
        n_estimators=5,
        max_depth=3,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    model.fit(X, treatment, y)
    fi = model.feature_importances_
    assert fi.shape == (X.shape[1],)
    assert np.isclose(fi.sum(), 1.0)
    # Uplift split gains are not a monotone impurity decrease, so per-feature
    # importances can be signed; they just have to be finite and averaged.
    assert np.isfinite(fi).all()


# ---------------------------------------------------------------------------
# Serialization + BaseEstimator conformance (issue #954)
# ---------------------------------------------------------------------------
# The kernel classes inherit SerializableLearner (save/load) and are sklearn
# BaseEstimators (get_params / clone) via BaseDecisionTree / ForestRegressor.
# Full check_estimator does not apply: fit takes (X, treatment, y), so sklearn's
# fit(X, y)-shaped checks cannot run. These cover the achievable conformance.


def _make_serialization_estimator(kind, **overrides):
    """Construct an unfitted kernel tree or forest with test-friendly defaults."""
    if kind == "tree":
        params = dict(
            criterion="KL",
            control_name=CONTROL_NAME,
            max_depth=3,
            min_samples_leaf=100,
            random_state=RANDOM_SEED,
        )
        params.update(overrides)
        return _KernelUpliftTreeClassifier(**params)
    params = dict(
        control_name=CONTROL_NAME,
        n_estimators=5,
        criterion="KL",
        max_depth=3,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    params.update(overrides)
    return _KernelUpliftRandomForestClassifier(**params)


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_kernel_uplift_save_load_round_trip(kind, tmp_path):
    """save() then load() restores an estimator with identical predictions."""
    X, treatment, y = _make_binary_feature_data()
    est = _make_serialization_estimator(kind).fit(X, treatment, y)
    path = str(tmp_path / f"{kind}.causalml")
    est.save(path)

    loaded = type(est).load(path)
    assert type(loaded) is type(est)
    assert_array_almost_equal(
        np.asarray(loaded.predict(X)), np.asarray(est.predict(X)), decimal=12
    )


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_kernel_uplift_unfitted_save_raises(kind, tmp_path):
    """Saving before fit is rejected."""
    est = _make_serialization_estimator(kind)
    with pytest.raises(ValueError):
        est.save(str(tmp_path / "unfitted.causalml"))


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_kernel_uplift_get_params_clone_round_trip(kind):
    """clone() reproduces the constructor params exactly (BaseEstimator)."""
    est = _make_serialization_estimator(kind)
    assert clone(est).get_params() == est.get_params()


def test_kernel_uplift_load_class_mismatch_raises(tmp_path):
    """Loading a saved tree as the forest class is rejected."""
    X, treatment, y = _make_binary_feature_data()
    tree = _make_serialization_estimator("tree").fit(X, treatment, y)
    path = str(tmp_path / "tree.causalml")
    tree.save(path)
    with pytest.raises(ValueError):
        _KernelUpliftRandomForestClassifier.load(path)


def test_kernel_uplift_init_stores_args_verbatim():
    """__init__ stores its arguments unchanged (sklearn get_params / clone contract).

    In particular ``criterion="IDDP"`` must not mutate the stored ``honesty`` --
    IDDP forces the honest approach in fit(), not in the constructor (see
    ``test_kernel_uplift_iddp_forces_honesty``).
    """
    args = dict(
        criterion="IDDP",
        control_name="ctrl",
        max_depth=7,
        min_samples_leaf=42,
        min_samples_treatment=3,
        n_reg=5,
        normalization=False,
        honesty=False,
        estimation_sample_size=0.3,
        random_state=123,
    )
    params = _KernelUpliftTreeClassifier(**args).get_params()
    for name, value in args.items():
        assert params[name] == value
