# distutils: language = c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

"""Causal tree builders: thin subclasses of the shared group tree builders.

The only causal-specific behavior is the per-group minimum-size stopping rule --
a split is inadmissible unless the control group and every treatment group are
large enough. Everything else (the depth-/best-first traversal) lives in
``_tree/_group_builder``.
"""

from .._tree._group_builder cimport (
    GroupDepthFirstTreeBuilder,
    GroupBestFirstTreeBuilder,
)
from .._tree._splitter cimport Splitter
from .._tree._typedefs cimport intp_t, int32_t, int64_t, float64_t
from .._tree._group_criterion cimport CausalRegressionCriterion


cdef inline bint _causal_extra_is_leaf(
    Splitter splitter,
    intp_t min_samples_split,
    intp_t min_samples_leaf,
    intp_t min_group_samples,
) noexcept nogil:
    """Per-group minimum-size stopping rules for causal trees.

    Mirrors the inline group-size checks in the pre-refactor causal builder: the
    per-group mean treatment count and the control count must each clear
    ``min_samples_split // groups_count`` and ``min_samples_leaf``, and the
    smallest group must clear ``min_group_samples``.
    """
    cdef int64_t tr_count_mean
    cdef int32_t ct_count
    cdef int32_t groups_count
    cdef int32_t min_size
    (<CausalRegressionCriterion> splitter.criterion).get_group_stats(
        &groups_count, &tr_count_mean, &ct_count, &min_size)
    return (tr_count_mean < min_samples_split // groups_count or
            ct_count < min_samples_split // groups_count or
            tr_count_mean < min_samples_leaf or
            ct_count < min_samples_leaf or
            min_size < min_group_samples)


cdef class DepthFirstCausalTreeBuilder(GroupDepthFirstTreeBuilder):
    """Build a causal tree depth-first (with per-group minimum-size stopping)."""

    cdef intp_t min_group_samples

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease,
                  intp_t min_group_samples):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_group_samples = min_group_samples

    cdef bint _node_is_leaf_extra(self, Splitter splitter) noexcept nogil:
        return _causal_extra_is_leaf(
            splitter, self.min_samples_split, self.min_samples_leaf,
            self.min_group_samples)


cdef class BestFirstCausalTreeBuilder(GroupBestFirstTreeBuilder):
    """Build a causal tree best-first (with per-group minimum-size stopping)."""

    cdef intp_t min_group_samples

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf,  min_weight_leaf,
                  intp_t max_depth, intp_t max_leaf_nodes,
                  float64_t min_impurity_decrease, intp_t min_group_samples):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_group_samples = min_group_samples

    cdef bint _node_is_leaf_extra(self, Splitter splitter) noexcept nogil:
        return _causal_extra_is_leaf(
            splitter, self.min_samples_split, self.min_samples_leaf,
            self.min_group_samples)
