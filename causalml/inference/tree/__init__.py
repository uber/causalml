from .causal.causaltree import CausalTreeRegressor
from .causal.causalforest import CausalRandomForestRegressor
from .plot import uplift_tree_string, uplift_tree_plot, plot_dist_tree_leaves_values
from .uplift import DecisionTree, UpliftTreeClassifier, UpliftRandomForestClassifier
from .utils import (
    cat_group,
    cat_transform,
    cv_fold_index,
    cat_continuous,
    kpi_transform,
    get_tree_leaves_mask,
)

from causalml.inference.serialization import SerializableLearner


# Inject serialization into the Cython uplift classes.
# These are regular Python classes defined in a .pyx file, so we can
# add methods at import time without recompiling.


def _uplift_tree_is_fitted(self):
    """UpliftTreeClassifier is fitted when fitted_uplift_tree is not None."""
    return getattr(self, "fitted_uplift_tree", None) is not None


def _uplift_forest_is_fitted(self):
    """UpliftRandomForestClassifier is fitted when uplift_forest exists."""
    return hasattr(self, "uplift_forest")


# Patch UpliftTreeClassifier
UpliftTreeClassifier._is_fitted = _uplift_tree_is_fitted
UpliftTreeClassifier.save = SerializableLearner.save
UpliftTreeClassifier.load = classmethod(SerializableLearner.load.__func__)

# Patch UpliftRandomForestClassifier
UpliftRandomForestClassifier._is_fitted = _uplift_forest_is_fitted
UpliftRandomForestClassifier.save = SerializableLearner.save
UpliftRandomForestClassifier.load = classmethod(
    SerializableLearner.load.__func__
)
