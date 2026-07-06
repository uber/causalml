"""
Serialization mixin for causalml learners.

Provides save/load capabilities with version metadata and safety checks.
Any learner class can inherit from SerializableLearner to get a consistent
persistence API backed by joblib.
"""

import logging
import os
import platform
import warnings
from datetime import datetime, timezone

import joblib

logger = logging.getLogger("causalml")


class CausalMLVersionMismatchWarning(UserWarning):
    """Raised when a saved model was created with a different causalml version."""

    pass


def _get_causalml_version():
    """Return the installed causalml version string."""
    try:
        from importlib.metadata import version

        return version("causalml")
    except Exception:
        return "unknown"


class SerializableLearner:
    """Mixin that adds save/load to any causalml learner.

    When mixed into a learner class, it provides:
    - save(path): persist the fitted model with version metadata
    - load(path): restore a model from disk with safety checks

    Subclasses should override _is_fitted() if the default sklearn-based
    check does not apply (e.g. for non-sklearn learners).
    """

    def _is_fitted(self):
        """Check whether this learner has been fitted.

        The default tries sklearn's check_is_fitted. Override this in
        subclasses that signal fitted-ness differently.

        Returns:
            bool: True if the model appears to be fitted.
        """
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(self)
            return True
        except Exception:
            return False

    def save(self, path):
        """Save the fitted learner to disk.

        The file contains the full model state plus metadata for version
        tracking. Use the corresponding load() class method to restore it.

        Args:
            path (str): file path where the model will be saved.
                Convention is to use the .causalml extension, but any
                path works.

        Raises:
            ValueError: if the learner has not been fitted yet.
        """
        if not self._is_fitted():
            raise ValueError("Cannot save an unfitted model. Call fit() first.")

        metadata = {
            "causalml_version": _get_causalml_version(),
            "python_version": platform.python_version(),
            "learner_class": type(self).__name__,
            "learner_module": type(self).__module__,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        payload = {
            "model": self,
            "metadata": metadata,
        }

        # Make sure the directory exists.
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(payload, path)
        logger.info(
            "Model saved to %s (causalml %s)",
            path,
            metadata["causalml_version"],
        )

    @classmethod
    def load(cls, path):
        """Load a previously saved learner from disk.

        Checks the saved metadata against the current environment and warns
        if there is a version mismatch. Also verifies that the loaded model
        class matches the class you are loading from.

        Args:
            path (str): file path to the saved model.

        Returns:
            The restored learner instance, ready for predict().

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if the saved model class does not match.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at: {path}")

        payload = joblib.load(path)

        # Handle both the new format (dict with metadata) and raw joblib dumps.
        if isinstance(payload, dict) and "metadata" in payload:
            metadata = payload["metadata"]
            model = payload["model"]
        else:
            # Someone saved a raw model with joblib directly.
            warnings.warn(
                "Loaded a model without causalml metadata. "
                "Version compatibility cannot be verified.",
                CausalMLVersionMismatchWarning,
                stacklevel=2,
            )
            return payload

        # Check version mismatch.
        saved_version = metadata.get("causalml_version", "unknown")
        current_version = _get_causalml_version()
        if saved_version != current_version and saved_version != "unknown":
            warnings.warn(
                f"This model was saved with causalml {saved_version}, "
                f"but you are running causalml {current_version}. "
                f"Predictions may differ. Consider retraining the model.",
                CausalMLVersionMismatchWarning,
                stacklevel=2,
            )

        # Check class mismatch (unless loading via the generic load_learner).
        saved_class = metadata.get("learner_class", "")
        if cls is not SerializableLearner and saved_class != cls.__name__:
            raise ValueError(
                f"Class mismatch: the saved model is a {saved_class}, "
                f"but you are trying to load it as {cls.__name__}. "
                f"Use the correct class or use load_learner() instead."
            )

        logger.info(
            "Model loaded from %s (saved with causalml %s on %s)",
            path,
            saved_version,
            metadata.get("saved_at", "unknown"),
        )
        return model


def load_learner(path):
    """Load any saved causal learner without specifying the class.

    This is a convenience function that skips the class-match check,
    useful when you don't know which learner type was saved.

    Args:
        path (str): file path to the saved model.

    Returns:
        The restored learner instance.
    """
    return SerializableLearner.load(path)
