# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# distutils: language = c++

from .._tree._typedefs cimport float64_t
from ..causal._criterion cimport CausalRegressionCriterion, NodeState


cdef class UpliftClassificationCriterion(CausalRegressionCriterion):
    # Divergence contribution of a single treatment-vs-control pair. Overridden
    # by each concrete criterion (KL / ED / Chi).
    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil
    # Sum of pair divergences over all treatment groups for a node.
    cdef float64_t _node_divergence(self, NodeState node) noexcept nogil
