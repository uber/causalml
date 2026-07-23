"""Numerical parity tests for the kernel-backed uplift trees (epic #945, issue #946).

These assert that the experimental ``_KernelUpliftTreeClassifier`` -- which grows an
uplift tree on the shared ``_tree`` Cython kernel via the new
``UpliftClassificationCriterion`` -- reproduces the legacy ``UpliftTreeClassifier``
predictions exactly for the KL / ED / Chi criteria, with regularization,
normalization, honesty, and pruning all disabled.

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
import pytest
from numpy.testing import assert_array_almost_equal

from causalml.inference.tree.uplift import UpliftTreeClassifier
from causalml.inference.tree._uplift.uplifttree import _KernelUpliftTreeClassifier

from .const import RANDOM_SEED, CONTROL_NAME

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
