"""
Backward-compatible re-export of the serialization mixin.

The mixin now lives in causalml.inference.serialization so it can be
shared across all learner families (meta, tree, IV). This module keeps
the old import path working.
"""

from causalml.inference.serialization import (  # noqa: F401
    CausalMLVersionMismatchWarning,
    SerializableLearner,
    load_learner,
)
