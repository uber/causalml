"""Tests for model serialization (save/load) of causal meta-learners."""

import os
import tempfile
import warnings

import joblib
import numpy as np
import pytest
from xgboost import XGBRegressor, XGBClassifier

from causalml.dataset import synthetic_data, make_uplift_classification
from causalml.inference.meta import (
    BaseTRegressor,
    BaseTClassifier,
    BaseSRegressor,
    BaseSClassifier,
    BaseXRegressor,
    BaseRRegressor,
    BaseDRRegressor,
    XGBTRegressor,
    load_learner,
    CausalMLVersionMismatchWarning,
)

RANDOM_SEED = 42
N_SAMPLE = 500


@pytest.fixture(scope="module")
def regression_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(RANDOM_SEED)
    data = synthetic_data(mode=1, n=N_SAMPLE, p=5, sigma=0.5)
    y, X, treatment, _, _, e = data
    return X, treatment, y


@pytest.fixture(scope="module")
def classification_data():
    """Generate synthetic classification data for testing."""
    np.random.seed(RANDOM_SEED)
    data = make_uplift_classification(
        n_samples=N_SAMPLE,
        treatment_name=["control", "treatment1"],
        y_name="conversion",
        random_seed=RANDOM_SEED,
    )
    return data


@pytest.fixture
def tmp_path_file():
    """Provide a temporary file path that is cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".causalml", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Round-trip tests for each learner type
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test that save/load produces identical predictions for each learner."""

    def test_tregressor_round_trip(self, regression_data, tmp_path_file):
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X)

        learner.save(tmp_path_file)
        loaded = BaseTRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X=X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_sregressor_round_trip(self, regression_data, tmp_path_file):
        X, treatment, y = regression_data
        learner = BaseSRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X, treatment=treatment)

        learner.save(tmp_path_file)
        loaded = BaseSRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X=X, treatment=treatment)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_xregressor_round_trip(self, regression_data, tmp_path_file):
        X, treatment, y = regression_data
        learner = BaseXRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X)

        learner.save(tmp_path_file)
        loaded = BaseXRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X=X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_rregressor_round_trip(self, regression_data, tmp_path_file):
        X, treatment, y = regression_data
        learner = BaseRRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X)

        learner.save(tmp_path_file)
        loaded = BaseRRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X=X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_drregressor_round_trip(self, regression_data, tmp_path_file):
        X, treatment, y = regression_data
        learner = BaseDRRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X)

        learner.save(tmp_path_file)
        loaded = BaseDRRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X=X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_xgbtregressor_round_trip(self, regression_data, tmp_path_file):
        """Test the convenience XGBTRegressor subclass."""
        X, treatment, y = regression_data
        learner = XGBTRegressor()
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X)

        learner.save(tmp_path_file)
        loaded = XGBTRegressor.load(tmp_path_file)
        preds_after = loaded.predict(X=X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_generic_load_learner(self, regression_data, tmp_path_file):
        """Test the generic load_learner() function."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        preds_before = learner.predict(X=X)

        learner.save(tmp_path_file)
        loaded = load_learner(tmp_path_file)
        preds_after = loaded.predict(X=X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# Safety and error handling tests
# ---------------------------------------------------------------------------


class TestSafetyChecks:
    """Test error handling and safety mechanisms."""

    def test_save_unfitted_raises(self, tmp_path_file):
        """Saving an unfitted model should raise ValueError."""
        learner = BaseTRegressor(learner=XGBRegressor())
        with pytest.raises(ValueError, match="Cannot save an unfitted model"):
            learner.save(tmp_path_file)

    def test_load_missing_file_raises(self):
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No saved model found"):
            BaseTRegressor.load("/tmp/this_file_does_not_exist_12345.causalml")

    def test_class_mismatch_raises(self, regression_data, tmp_path_file):
        """Loading a T-learner as an S-learner should raise ValueError."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        learner.save(tmp_path_file)

        with pytest.raises(ValueError, match="Class mismatch"):
            BaseSRegressor.load(tmp_path_file)

    def test_version_mismatch_warning(self, regression_data, tmp_path_file):
        """Loading a model saved with a different version should warn."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        learner.save(tmp_path_file)

        # Tamper with the version in the saved file.
        payload = joblib.load(tmp_path_file)
        payload["metadata"]["causalml_version"] = "0.0.1"
        joblib.dump(payload, tmp_path_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = BaseTRegressor.load(tmp_path_file)
            assert len(w) == 1
            assert issubclass(w[0].category, CausalMLVersionMismatchWarning)
            assert "0.0.1" in str(w[0].message)

    def test_no_warning_when_versions_match(self, regression_data, tmp_path_file):
        """No warning should be raised when versions match."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        learner.save(tmp_path_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = BaseTRegressor.load(tmp_path_file)
            version_warnings = [
                x for x in w if issubclass(x.category, CausalMLVersionMismatchWarning)
            ]
            assert len(version_warnings) == 0


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestMetadata:
    """Test that metadata is correctly stored and accessible."""

    def test_metadata_fields_present(self, regression_data, tmp_path_file):
        """Saved file should contain all expected metadata fields."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        learner.save(tmp_path_file)

        payload = joblib.load(tmp_path_file)
        metadata = payload["metadata"]

        assert "causalml_version" in metadata
        assert "python_version" in metadata
        assert "learner_class" in metadata
        assert "learner_module" in metadata
        assert "saved_at" in metadata

    def test_metadata_class_name_correct(self, regression_data, tmp_path_file):
        """The learner_class field should match the actual class name."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        learner.save(tmp_path_file)

        payload = joblib.load(tmp_path_file)
        assert payload["metadata"]["learner_class"] == "BaseTRegressor"

    def test_save_creates_directory(self, regression_data):
        """save() should create intermediate directories if they don't exist."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)

        path = "/tmp/causalml_test_nested/subdir/model.causalml"
        try:
            learner.save(path)
            assert os.path.exists(path)
        finally:
            import shutil

            shutil.rmtree("/tmp/causalml_test_nested", ignore_errors=True)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_overwrite_existing_file(self, regression_data, tmp_path_file):
        """Saving to an existing file should overwrite it without error."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)

        # Save twice to the same path.
        learner.save(tmp_path_file)
        learner.save(tmp_path_file)

        loaded = BaseTRegressor.load(tmp_path_file)
        preds = loaded.predict(X=X)
        assert preds.shape[0] == X.shape[0]

    def test_save_load_with_bootstrap_ensemble(self, regression_data, tmp_path_file):
        """A model with bootstrap_models_ should save and load correctly."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(
            X=X,
            treatment=treatment,
            y=y,
            store_bootstraps=True,
            n_bootstraps=5,
            bootstrap_size=200,
            random_state=0,
        )

        assert learner.bootstrap_models_ is not None
        assert len(learner.bootstrap_models_) == 5

        learner.save(tmp_path_file)
        loaded = BaseTRegressor.load(tmp_path_file)

        assert loaded.bootstrap_models_ is not None
        assert len(loaded.bootstrap_models_) == 5

    def test_multiple_save_load_cycles(self, regression_data, tmp_path_file):
        """Model should survive multiple save/load cycles without degradation."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)
        original_preds = learner.predict(X=X)

        # Save, load, save again, load again.
        for _ in range(3):
            learner.save(tmp_path_file)
            learner = BaseTRegressor.load(tmp_path_file)

        final_preds = learner.predict(X=X)
        np.testing.assert_array_almost_equal(original_preds, final_preds)

    def test_load_raw_joblib_fallback(self, regression_data, tmp_path_file):
        """Loading a raw joblib dump (no metadata) should work with a warning."""
        X, treatment, y = regression_data
        learner = BaseTRegressor(learner=XGBRegressor(n_estimators=10, random_state=0))
        learner.fit(X=X, treatment=treatment, y=y)

        # Save the model directly with joblib (no metadata wrapper).
        joblib.dump(learner, tmp_path_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_learner(tmp_path_file)
            assert len(w) == 1
            assert "without causalml metadata" in str(w[0].message)

        preds = loaded.predict(X=X)
        assert preds.shape[0] == X.shape[0]
