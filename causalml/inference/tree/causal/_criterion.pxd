# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True


from sklearn.tree._criterion cimport RegressionCriterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t

IF SKLEARN_VERSION < 12:
    # If SKLearn < 1.2, use DOUBLE_t* as DOUBLE_ARRAY_t
    ctypedef DOUBLE_t* DOUBLE_ARRAY_t
ELSE:
    # If SKLearn >= 1.2, use const DOUBLE_t[:] as DOUBLE_ARRAY_t
    ctypedef const DOUBLE_t[:] DOUBLE_ARRAY_t

IF SKLEARN_VERSION < 13:
    # If SKLearn < 1.3, use DOUBLE_t* as DOUBLE_ARRAY_t
    ctypedef SIZE_t* SIZE_ARRAY_t
ELSE:
    # Else use const SIZE_t[:] memory view
    ctypedef const SIZE_t[:] SIZE_ARRAY_t


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
