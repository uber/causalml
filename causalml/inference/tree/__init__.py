import warnings

from .causal.causaltree import CausalTreeRegressor
from .causal.causalforest import CausalRandomForestRegressor
from .plot import uplift_tree_string, uplift_tree_plot, plot_dist_tree_leaves_values
from ._uplift.uplifttree import UpliftTreeClassifier, _UpliftTreeNode
from ._uplift.upliftforest import UpliftRandomForestClassifier
from .utils import (
    cat_group,
    cat_transform,
    cv_fold_index,
    cat_continuous,
    kpi_transform,
    get_tree_leaves_mask,
)

# The public uplift classes are now kernel-backed subclasses that inherit
# ``save`` / ``load`` and sklearn ``get_params`` / ``clone`` directly (issue
# #955), so the historical import-time monkey-patch is gone.


def __getattr__(name):
    """Deprecation shim for the removed legacy ``DecisionTree`` export.

    ``DecisionTree`` was the pure-Python uplift tree node from the deleted
    ``uplift.pyx`` (issue #955). The kernel-backed trees use the minimal
    ``_UpliftTreeNode`` node instead; keep the name importable with a
    ``DeprecationWarning`` for one release.
    """
    if name == "DecisionTree":
        warnings.warn(
            "`causalml.inference.tree.DecisionTree` is deprecated and will be "
            "removed in a future release; the kernel-backed uplift trees use an "
            "internal node type. It now aliases the minimal plot node.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _UpliftTreeNode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
