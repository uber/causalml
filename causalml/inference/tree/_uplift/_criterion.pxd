# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# distutils: language = c++

from libcpp.vector cimport vector

from .._tree._typedefs cimport intp_t, float64_t
from ..causal._criterion cimport CausalRegressionCriterion, NodeState


cdef class UpliftClassificationCriterion(CausalRegressionCriterion):
    # Rzepakowski et al. (2012) parent-shrinkage regularization. ``n_reg`` is the
    # weight (in sample-size units) of the parent node's influence on a child;
    # ``min_samples_treatment`` is both the shrinkage threshold and the minimum
    # per-group size a candidate split's children must have to be admissible.
    cdef public float64_t n_reg
    cdef public intp_t min_samples_treatment

    # Rzepakowski et al. (2012) normalization: when set, the split gain is divided
    # by the treatment-vs-control distribution divergence ``_norm_factor`` (issue
    # #948). Not applied by criteria whose ``_normalizable`` returns 0 (e.g. CTS).
    cdef public bint normalization

    # Regularized positive-outcome probabilities of the *parent* node, threaded
    # down by the builder before each ``node_split`` (control at index 0). Only
    # valid when ``has_parent`` is non-zero (the root has no parent).
    cdef bint has_parent
    cdef vector[float64_t] parent_summary_p

    # Scratch buffers (size n_outputs) for the regularized node / child summaries.
    # ``_buf_left`` and ``_buf_right`` coexist so criteria that need both children
    # at once (IT / CIT) can read them together.
    cdef vector[float64_t] _buf_node
    cdef vector[float64_t] _buf_left
    cdef vector[float64_t] _buf_right

    # Divergence contribution of a single treatment-vs-control pair. Overridden
    # by each concrete criterion (KL / ED / Chi).
    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil
    # Sum of pair divergences over all treatment groups for a probability vector.
    cdef float64_t _divergence_of(self, float64_t* p) noexcept nogil
    # Per-node split metric: gain = p*M(left) + (1-p)*M(right) - M(node). Defaults
    # to ``_divergence_of``; CTS overrides it with the max over groups.
    cdef float64_t _node_metric(self, float64_t* p) noexcept nogil
    # Split gain from the regularized node / left / right summaries and the left
    # fraction ``p``. Defaults to ``p*M(left)+(1-p)*M(right)-M(node)`` (KL/ED/Chi/
    # CTS); DDP/IT/CIT override it with their own two-class gain forms.
    cdef float64_t _split_gain(self, float64_t* s_node, float64_t* s_left, float64_t* s_right, float64_t p) noexcept nogil
    # Whether ``_norm_factor`` normalization applies (0 for CTS/DDP/IT/CIT).
    cdef bint _normalizable(self) noexcept nogil
    # Rzepakowski normalization factor from the raw node / one-child group counts
    # (the child matching legacy ``arr_normI``'s asymmetric "left" = X >= value).
    # ``gain`` is the raw (pre-normalization) split gain; the KL/ED/Chi factor
    # ignores it, IDDP uses it (``currentDivergence = 2*(gain+1)/3``).
    cdef float64_t _norm_factor(self, float64_t gain) noexcept nogil
    # Regularized P(Y=1|T=g) of ``node``, shrunk toward ``target_p`` (when
    # ``has_target``), written into ``dest``.
    cdef void _reg_summary(self, NodeState node, float64_t* target_p, bint has_target, float64_t* dest) noexcept nogil
    # Builder API: set the parent summary before ``node_split``; read this node's
    # regularized summary back out after ``node_value`` to thread to its children.
    cdef void set_parent_summary(self, float64_t* src, bint has_parent) noexcept nogil
    cdef void compute_reg_summary(self, float64_t* dest) noexcept nogil
