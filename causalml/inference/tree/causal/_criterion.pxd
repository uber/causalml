# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True


from .._tree._criterion cimport RegressionCriterion
from .._tree._criterion cimport int32_t, intp_t, float64_t


cdef struct NodeInfo:
    double count        # the number of obs
    double tr_count     # the number of treatment obs
    double ct_count     # the number of control obs
    double tr_y_sum     # the sum of outcomes among treatment obs
    double ct_y_sum     # the sum of outcomes among control obs
    double y_sq_sum     # the squared sum of outcomes
    double tr_y_sq_sum  # the squared sum of outcomes among treatment obs
    double ct_y_sq_sum  # the squared sum of outcomes among control obs
    double split_metric # Additional split metric for t-test criterion

cdef struct SplitState:
    NodeInfo node   # current node state
    NodeInfo right  # right split state
    NodeInfo left   # left split state
