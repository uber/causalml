# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
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
# The treatment-group criterion base now lives in the shared ``_tree`` kernel.
# The concrete causal criteria (StandardMSE / CausalMSE / TTest) below subclass
# it and read the shared control-group index.
from .._tree._group_criterion cimport (
    NodeState,
    NodeSplitState,
    CausalRegressionCriterion,
    CONTROL_GROUP_IDX,
)
