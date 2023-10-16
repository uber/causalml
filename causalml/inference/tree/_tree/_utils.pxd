# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True

import numpy as np
cimport numpy as cnp
from ._tree cimport Node

ctypedef cnp.npy_float32 DTYPE_t          # Type of X
ctypedef cnp.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Node**)
    (StackRecord*)
    (FrontierRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *


cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) nogil


cdef double rand_uniform(double low, double high,
                         UINT32_t* random_state) nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)


cdef double log(double x) nogil

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features

# =============================================================================
# PriorityHeap data structure
# =============================================================================

# A record on the frontier for best-first tree growing
cdef struct FrontierRecord:
    # Record of information of a Node, the frontier for a split. Those records are
    # maintained in a heap to access the Node with the best improvement in impurity,
    # allowing growing trees greedily on this improvement.
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t pos
    SIZE_t depth
    bint is_leaf
    double impurity
    double impurity_left
    double impurity_right
    double improvement
