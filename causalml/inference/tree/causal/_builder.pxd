# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True

from sklearn.tree._tree cimport Node, Tree, TreeBuilder
from sklearn.tree._tree cimport Splitter, SplitRecord
from sklearn.tree._utils cimport StackRecord, Stack
from sklearn.tree._utils cimport PriorityHeapRecord, PriorityHeap
from sklearn.tree._tree cimport SIZE_t, DOUBLE_t


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