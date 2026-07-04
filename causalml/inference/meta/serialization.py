"""Model serialization for causal meta-learners.

Provides save/load functionality with version metadata to prevent
stale model execution in production environments.
"""

import json
import logging
import os
import platform
import sys
import warnings
from datetime import datetime, timezone

import joblib

logger = logging.getLogger("causalml")


def _get_causalml_version():
    """Get the installed causalml version string."""
    try:
        from importlib.metadata import version

        return version("causalml")
    except Exception:
        return "unknown"


class CausalMLVersionMismatchWarning(UserWarning):
    """Raised when loading a model saved with a different causalml version."""

    pass


class SerializableLearner:
    """Mixin that adds save/load capabilities to causal meta-learners.

    When mixed into a learner class, it provides:
    - save(path): persist the fitted model to disk with version metadata
    - load(path): class method to restore a model from disk with safety checks

    The saved file is a joblib dump containing the model object and a metadata
    dict with the causalml version, python version, class name, and timestamp.
    This metadata is checked on load to warn about potential incompatibilities.
    """

    def save(self, path):
        """Save the fitted learner to disk.

        The file contains the full model state plus metadata for version
        tracking. Use the corresponding load() class method to restore it.

        Args:
            path (str): file path where the model will be saved.
                Convention is to use the .causalml extension, but any
                path works.

        Raises:
            ValueError: if the learner has not been fitted yet (no t_groups).
        """
        # Basic check that the model has been fitted.
        if not hasattr(self, "t_groups"):
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

        This method checks the saved metadata against the current environment
        and warns if there is a version mismatch. It also verifies that the
        loaded model class matches the class you are loading from.

        Args:
            path (str): file path to the saved model.

        Returns:
            The restored learner instance, ready for predict().

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if the saved model class does not match the loading class.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at: {path}")

        payload = joblib.load(path)

        # Handle both the new format (dict with metadata) and raw joblib dumps.
        if isinstance(payload, dict) and "metadata" in payload:
            metadata = payload["metadata"]
            model = payload["model"]
        else:
            # Fallback: someone saved a raw model with joblib directly.
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
