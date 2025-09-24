# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True
# distutils: language = c++

from libc.math cimport fabs
from libc.math cimport isnan
from libc.math cimport sqrt
from libc.limits cimport INT_MAX
from libc.string cimport memset
from libc.string cimport memcpy
from libc.stdio cimport printf

from libcpp.vector cimport vector

from .._tree._typedefs cimport int32_t, int64_t, intp_t, float64_t
from .._tree._criterion cimport RegressionCriterion


cdef class NodeState:
    cdef public vector[float64_t] count_1d      
    cdef public vector[float64_t] y_sum_1d
    cdef public vector[float64_t] y_sq_sum_1d
    cdef public int32_t control_idx
    cdef public int32_t control_total
    cdef public int32_t treatment_total
    cdef public int32_t groups_total
    # Criterion-specific variables
    cdef public float64_t split_metric
    """
    NodeState cython class tracks statistics of a control group and multiple test groups 
    
    count_1d:       vector[float64_t],    the number of observations for a particular group
    y_sum_1d:       vector[float64_t],    the sum of y-s for a particular group
    y_sq_sum_1d:    vector[float64_t],    the sum of squared y-s for a particular group
    control_idx:    int32_t,              control group index
    control_total   int32_t,              total number of observations for a control group
    treatment_total int32_t,              total number of observations for treatment groups
    treatment_min   int32_t,    
    groups_total    int32_t,              total number of groups
    split_metric:   float64_t,            split metric for TTest criterion
    """

    cdef int32_t reset(self, intp_t n_outputs) except -1 nogil
    cdef int32_t update_counters(self) except -1 nogil
    cdef int32_t copy_from_state(self, NodeState state) except -1 nogil
    cdef int32_t increment_count(self, int32_t group_idx, float64_t value) except -1 nogil
    cdef int32_t increment_y_sum(self, int32_t group_idx, float64_t value) except -1 nogil
    cdef int32_t increment_y_sq_sum(self, int32_t group_idx, float64_t value) except -1 nogil
    cdef float64_t outcome_mean(self, int32_t group_idx) noexcept nogil
    cdef float64_t outcome_var(self, int32_t group_idx) noexcept nogil
    cdef float64_t effect(self, int32_t treatment_idx)  noexcept nogil


cdef class NodeSplitState:
    cdef public NodeState node
    cdef public NodeState right
    cdef public NodeState left

    """
    NodeSplitState cython class tracks statistics for the current node and potential left and right splits.

    node:    NodeState,    current node statistics
    right:   NodeState,    right split statistics
    left:    NodeState,    left split statistics
    """

    cdef int32_t reset_nodes(self, intp_t n_outputs) except -1 nogil
    """
    Prepare vectors or set existing ones to zero for each NodeState.
    """


cdef class CausalRegressionCriterion(RegressionCriterion):

    cdef public NodeSplitState state
    cdef public float64_t groups_penalty

    cdef int get_group_stats(
        self,
        int32_t* groups_count,
        int64_t* tr_count_mean,
        int32_t* ct_count,
        int32_t* min_size_among_groups) except -1 nogil
    cdef float64_t get_groups_penalty(self, NodeState node) noexcept nogil