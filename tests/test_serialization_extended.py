"""Tests for extended serialization: CausalTree, CausalForest, UpliftTree, UpliftForest, IV/DRIV."""

import os
import tempfile
import warnings

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_regression

from causalml.dataset import synthetic_data, make_uplift_classification
from causalml.inference.serialization import (
    SerializableLearner,
    load_learner,
    CausalMLVersionMismatchWarning,
)
from causalml.inference.tree import (
    CausalTreeRegressor,
    CausalRandomForestRegressor,
    UpliftTreeClassifier,
    UpliftRandomForestClassifier,
)
from causalml.inference.iv.iv_regression import IVRegressor

RANDOM_SEED = 42
N_SAMPLE = 300


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_path_file():
    """Provide a temporary file path that is cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".causalml", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture(scope="module")
def causal_tree_data():
    """Generate data suitable for CausalTreeRegressor."""
    np.random.seed(RANDOM_SEED)
    n = N_SAMPLE
    X = np.random.randn(n, 5)
    # CausalForest requires integer treatment labels.
    treatment = np.random.choice([0, 1], size=n)
    y = X[:, 0] + 2.0 * treatment + np.random.randn(n) * 0.5
    return X, treatment, y


CONTROL_NAME = 0


@pytest.fixture(scope="module")
def uplift_data():
    """Generate data suitable for UpliftTreeClassifier."""
    np.random.seed(RANDOM_SEED)
    df, _ = make_uplift_classification(
        n_samples=N_SAMPLE,
        treatment_name=["control", "treatment1"],
        y_name="conversion",
        random_seed=RANDOM_SEED,
    )
    return df


@pytest.fixture(scope="module")
def iv_data():
    """Generate data for IVRegressor (instrument + endogenous treatment)."""
    np.random.seed(RANDOM_SEED)
    n = N_SAMPLE
    X = np.random.randn(n, 3)
    # Instrument
    z = np.random.randn(n)
    # Endogenous treatment correlated with instrument
    treatment = 0.5 * z + np.random.randn(n) * 0.3
    # Outcome depends on X and treatment
    y = X[:, 0] + 2.0 * treatment + np.random.randn(n) * 0.5
    return X, treatment, y, z


# ---------------------------------------------------------------------------
# Round-trip tests for Group A learners
# ---------------------------------------------------------------------------


class TestCausalTreeRoundTrip:
    """Save/load round-trip for CausalTreeRegressor."""

    def test_round_trip_predictions_match(self, causal_tree_data, tmp_path_file):
        X, treatment, y = causal_tree_data
        tree = CausalTreeRegressor(control_name=CONTROL_NAME, random_state=RANDOM_SEED)
        tree.fit(X=X, treatment=treatment, y=y)
        preds_before = tree.predict(X)

        tree.save(tmp_path_file)
        loaded = CausalTreeRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_generic_load_learner(self, causal_tree_data, tmp_path_file):
        X, treatment, y = causal_tree_data
        tree = CausalTreeRegressor(control_name=CONTROL_NAME, random_state=RANDOM_SEED)
        tree.fit(X=X, treatment=treatment, y=y)
        tree.save(tmp_path_file)

        loaded = load_learner(tmp_path_file)
        assert isinstance(loaded, CausalTreeRegressor)

    def test_unfitted_save_raises(self, tmp_path_file):
        tree = CausalTreeRegressor(control_name=CONTROL_NAME)
        with pytest.raises(ValueError, match="Cannot save an unfitted model"):
            tree.save(tmp_path_file)

    def test_class_mismatch_raises(self, causal_tree_data, tmp_path_file):
        X, treatment, y = causal_tree_data
        tree = CausalTreeRegressor(control_name=CONTROL_NAME, random_state=RANDOM_SEED)
        tree.fit(X=X, treatment=treatment, y=y)
        tree.save(tmp_path_file)

        with pytest.raises(ValueError, match="Class mismatch"):
            CausalRandomForestRegressor.load(tmp_path_file)


class TestCausalForestRoundTrip:
    """Save/load round-trip for CausalRandomForestRegressor."""

    def test_round_trip_predictions_match(self, causal_tree_data, tmp_path_file):
        X, treatment, y = causal_tree_data
        forest = CausalRandomForestRegressor(
            n_estimators=5, control_name=CONTROL_NAME, random_state=RANDOM_SEED
        )
        forest.fit(X=X, treatment=treatment, y=y)
        preds_before = forest.predict(X)

        forest.save(tmp_path_file)
        loaded = CausalRandomForestRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_unfitted_save_raises(self, tmp_path_file):
        forest = CausalRandomForestRegressor(n_estimators=5, control_name=CONTROL_NAME)
        with pytest.raises(ValueError, match="Cannot save an unfitted model"):
            forest.save(tmp_path_file)

    def test_metadata_class_correct(self, causal_tree_data, tmp_path_file):
        X, treatment, y = causal_tree_data
        forest = CausalRandomForestRegressor(
            n_estimators=5, control_name=CONTROL_NAME, random_state=RANDOM_SEED
        )
        forest.fit(X=X, treatment=treatment, y=y)
        forest.save(tmp_path_file)

        payload = joblib.load(tmp_path_file)
        assert payload["metadata"]["learner_class"] == "CausalRandomForestRegressor"


class TestUpliftTreeRoundTrip:
    """Save/load round-trip for UpliftTreeClassifier."""

    def test_round_trip_predictions_match(self, uplift_data, tmp_path_file):
        df = uplift_data
        feature_cols = [
            c for c in df.columns if c not in ("treatment_group_key", "conversion")
        ]
        X = df[feature_cols].values
        treatment = df["treatment_group_key"].values
        y = df["conversion"].values

        tree = UpliftTreeClassifier(
            control_name="control", max_depth=3, min_samples_leaf=20
        )
        tree.fit(X, treatment, y)
        preds_before = tree.predict(X)

        tree.save(tmp_path_file)
        loaded = UpliftTreeClassifier.load(tmp_path_file)
        preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_unfitted_save_raises(self, tmp_path_file):
        tree = UpliftTreeClassifier(control_name="control")
        with pytest.raises(ValueError, match="Cannot save an unfitted model"):
            tree.save(tmp_path_file)

    def test_generic_load_learner(self, uplift_data, tmp_path_file):
        df = uplift_data
        feature_cols = [
            c for c in df.columns if c not in ("treatment_group_key", "conversion")
        ]
        X = df[feature_cols].values
        treatment = df["treatment_group_key"].values
        y = df["conversion"].values

        tree = UpliftTreeClassifier(
            control_name="control", max_depth=3, min_samples_leaf=20
        )
        tree.fit(X, treatment, y)
        tree.save(tmp_path_file)

        loaded = load_learner(tmp_path_file)
        assert isinstance(loaded, UpliftTreeClassifier)


class TestUpliftForestRoundTrip:
    """Save/load round-trip for UpliftRandomForestClassifier."""

    def test_round_trip_predictions_match(self, uplift_data, tmp_path_file):
        df = uplift_data
        feature_cols = [
            c for c in df.columns if c not in ("treatment_group_key", "conversion")
        ]
        X = df[feature_cols].values
        treatment = df["treatment_group_key"].values
        y = df["conversion"].values

        forest = UpliftRandomForestClassifier(
            control_name="control",
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        )
        forest.fit(X, treatment, y)
        preds_before = forest.predict(X)

        forest.save(tmp_path_file)
        loaded = UpliftRandomForestClassifier.load(tmp_path_file)
        preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_unfitted_save_raises(self, tmp_path_file):
        forest = UpliftRandomForestClassifier(control_name="control", n_estimators=5)
        with pytest.raises(ValueError, match="Cannot save an unfitted model"):
            forest.save(tmp_path_file)


class TestIVRegressorRoundTrip:
    """Save/load round-trip for IVRegressor."""

    def test_round_trip_predictions_match(self, iv_data, tmp_path_file):
        X, treatment, y, z = iv_data
        model = IVRegressor()
        model.fit(X, treatment, y, z)
        ate_before, se_before = model.predict()

        model.save(tmp_path_file)
        loaded = IVRegressor.load(tmp_path_file)
        ate_after, se_after = loaded.predict()

        assert ate_before == pytest.approx(ate_after)
        assert se_before == pytest.approx(se_after)

    def test_unfitted_save_raises(self, tmp_path_file):
        model = IVRegressor()
        with pytest.raises(ValueError, match="Cannot save an unfitted model"):
            model.save(tmp_path_file)

    def test_generic_load_learner(self, iv_data, tmp_path_file):
        X, treatment, y, z = iv_data
        model = IVRegressor()
        model.fit(X, treatment, y, z)
        model.save(tmp_path_file)

        loaded = load_learner(tmp_path_file)
        assert isinstance(loaded, IVRegressor)

    def test_metadata_fields_present(self, iv_data, tmp_path_file):
        X, treatment, y, z = iv_data
        model = IVRegressor()
        model.fit(X, treatment, y, z)
        model.save(tmp_path_file)

        payload = joblib.load(tmp_path_file)
        metadata = payload["metadata"]
        assert "causalml_version" in metadata
        assert "python_version" in metadata
        assert metadata["learner_class"] == "IVRegressor"


# ---------------------------------------------------------------------------
# Cross-family edge cases
# ---------------------------------------------------------------------------


class TestCrossFamilyEdgeCases:
    """Edge cases that span multiple learner families."""

    def test_load_tree_as_forest_raises(self, causal_tree_data, tmp_path_file):
        """Loading a CausalTree file as CausalForest should fail."""
        X, treatment, y = causal_tree_data
        tree = CausalTreeRegressor(control_name=CONTROL_NAME, random_state=RANDOM_SEED)
        tree.fit(X=X, treatment=treatment, y=y)
        tree.save(tmp_path_file)

        with pytest.raises(ValueError, match="Class mismatch"):
            CausalRandomForestRegressor.load(tmp_path_file)

    def test_load_iv_as_tree_raises(self, iv_data, tmp_path_file):
        """Loading an IV model as a CausalTree should fail."""
        X, treatment, y, z = iv_data
        model = IVRegressor()
        model.fit(X, treatment, y, z)
        model.save(tmp_path_file)

        with pytest.raises(ValueError, match="Class mismatch"):
            CausalTreeRegressor.load(tmp_path_file)

    def test_version_mismatch_warning(self, causal_tree_data, tmp_path_file):
        """Version mismatch should produce a warning, not an error."""
        X, treatment, y = causal_tree_data
        tree = CausalTreeRegressor(control_name=CONTROL_NAME, random_state=RANDOM_SEED)
        tree.fit(X=X, treatment=treatment, y=y)
        tree.save(tmp_path_file)

        # Tamper with the saved version.
        payload = joblib.load(tmp_path_file)
        payload["metadata"]["causalml_version"] = "0.0.0"
        joblib.dump(payload, tmp_path_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = CausalTreeRegressor.load(tmp_path_file)
            version_warnings = [
                x for x in w if issubclass(x.category, CausalMLVersionMismatchWarning)
            ]
            assert len(version_warnings) == 1
            assert "0.0.0" in str(version_warnings[0].message)

    def test_multiple_save_load_cycles(self, iv_data, tmp_path_file):
        """Model should survive multiple save/load cycles."""
        X, treatment, y, z = iv_data
        model = IVRegressor()
        model.fit(X, treatment, y, z)
        ate_original, se_original = model.predict()

        for _ in range(3):
            model.save(tmp_path_file)
            model = IVRegressor.load(tmp_path_file)

        ate_final, se_final = model.predict()
        assert ate_original == pytest.approx(ate_final)
        assert se_original == pytest.approx(se_final)

    def test_backward_compat_import_path(self):
        """The old import path should still work."""
        from causalml.inference.meta.serialization import (
            SerializableLearner as SL_old,
            load_learner as ll_old,
        )
        from causalml.inference.serialization import (
            SerializableLearner as SL_new,
            load_learner as ll_new,
        )

        assert SL_old is SL_new
        assert ll_old is ll_new
