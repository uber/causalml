# distutils: language = c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

"""Uplift tree builders: thin subclasses of the shared group tree builders.

The only uplift-specific behavior is **parent-summary threading**. The
regularization in :class:`UpliftClassificationCriterion` shrinks each node toward
its parent's *regularized* ``P(Y=1|T=g)``, which is not derivable from the raw
leaf values stored in ``tree.value`` (kept raw for prediction parity with the
legacy tree). The builder therefore keeps a ``node_id -> regularized summary``
map: the parent summary is threaded into the criterion before each split search
(``_before_node_split``) and this node's summary is recorded after its value is
computed (``_after_node_value``), for its children to shrink toward.

The stock traversal (depth-/best-first) lives in ``_tree/_group_builder``. Note
the shared builders leaf a node on the stock ``impurity <= EPSILON`` rule in
best-first; for uplift ``node_impurity`` is a constant placeholder (``1.0``) so
that rule never fires, and it is absent from depth-first -- growth is driven by
the split gain, matching the legacy tree. Only the depth-first builder is
reachable from the public uplift API (``_KernelUpliftTreeClassifier`` fixes
``max_leaf_nodes=None``); the best-first subclass is kept for symmetry with the
causal trees.
"""

from libcpp.map cimport map
from libcpp.vector cimport vector

from .._tree._group_builder cimport (
    GroupDepthFirstTreeBuilder,
    GroupBestFirstTreeBuilder,
)
from .._tree._splitter cimport Splitter
from .._tree._tree cimport Tree
from .._tree._typedefs cimport intp_t, float64_t
from ._criterion cimport UpliftClassificationCriterion


cdef intp_t _TREE_UNDEFINED = -2


cdef inline void _thread_parent_summary(
    map[intp_t, vector[float64_t]]& reg_map,
    Splitter splitter,
    intp_t parent,
) noexcept nogil:
    """Set the criterion's parent summary before the split search."""
    if parent == _TREE_UNDEFINED:
        (<UpliftClassificationCriterion> splitter.criterion).set_parent_summary(NULL, 0)
    else:
        (<UpliftClassificationCriterion> splitter.criterion).set_parent_summary(
            &reg_map[parent][0], 1)


cdef inline void _record_node_summary(
    map[intp_t, vector[float64_t]]& reg_map,
    vector[float64_t]& scratch,
    Splitter splitter,
    intp_t node_id,
) noexcept nogil:
    """Store this node's regularized summary for its children to shrink toward."""
    (<UpliftClassificationCriterion> splitter.criterion).compute_reg_summary(&scratch[0])
    reg_map[node_id] = scratch


cdef class DepthFirstUpliftTreeBuilder(GroupDepthFirstTreeBuilder):
    """Build an uplift tree depth-first (with parent-summary threading)."""

    cdef map[intp_t, vector[float64_t]] reg_map
    cdef vector[float64_t] _node_summary

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cdef void _before_build(self, Tree tree) noexcept nogil:
        self.reg_map.clear()
        self._node_summary.resize(tree.n_outputs, 0.0)

    cdef void _before_node_split(self, Splitter splitter, intp_t parent) noexcept nogil:
        _thread_parent_summary(self.reg_map, splitter, parent)

    cdef void _after_node_value(self, Splitter splitter, intp_t node_id) noexcept nogil:
        _record_node_summary(self.reg_map, self._node_summary, splitter, node_id)


cdef class BestFirstUpliftTreeBuilder(GroupBestFirstTreeBuilder):
    """Build an uplift tree best-first (with parent-summary threading).

    Unreachable from the public API (see module docstring); kept for symmetry.
    """

    cdef map[intp_t, vector[float64_t]] reg_map
    cdef vector[float64_t] _node_summary

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf,  min_weight_leaf,
                  intp_t max_depth, intp_t max_leaf_nodes,
                  float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    cdef void _before_build(self, Tree tree) noexcept nogil:
        self.reg_map.clear()
        self._node_summary.resize(tree.n_outputs, 0.0)

    cdef void _before_node_split(self, Splitter splitter, intp_t parent) noexcept nogil:
        _thread_parent_summary(self.reg_map, splitter, parent)

    cdef void _after_node_value(self, Splitter splitter, intp_t node_id) noexcept nogil:
        _record_node_summary(self.reg_map, self._node_summary, splitter, node_id)
