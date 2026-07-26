# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# See _group_builder.pyx for implementation details.
from ._tree cimport Node, Tree, TreeBuilder
from ._splitter cimport Splitter, SplitRecord
from ._typedefs cimport intp_t, int32_t, int64_t, float32_t, float64_t
from ._tree cimport FrontierRecord, StackRecord
from ._tree cimport ParentInfo, _init_parent_record


cdef class GroupDepthFirstTreeBuilder(TreeBuilder):
    # Shared depth-first skeleton for treatment-group trees (causal + uplift).
    # Family-specific behavior is injected through the hooks below; all default
    # to no-ops so the base builder reproduces the stock sklearn traversal.
    cdef void _before_build(self, Tree tree) noexcept nogil
    cdef void _before_node_split(self, Splitter splitter, intp_t parent) noexcept nogil
    cdef void _after_node_value(self, Splitter splitter, intp_t node_id) noexcept nogil
    cdef bint _node_is_leaf_extra(self, Splitter splitter) noexcept nogil


cdef class GroupBestFirstTreeBuilder(TreeBuilder):
    cdef intp_t max_leaf_nodes

    cdef void _before_build(self, Tree tree) noexcept nogil
    cdef void _before_node_split(self, Splitter splitter, intp_t parent) noexcept nogil
    cdef void _after_node_value(self, Splitter splitter, intp_t node_id) noexcept nogil
    cdef bint _node_is_leaf_extra(self, Splitter splitter) noexcept nogil
    cdef int _add_split_node(
        self,
        Splitter splitter,
        Tree tree,
        intp_t start,
        intp_t end,
        bint is_first,
        bint is_left,
        Node* parent,
        intp_t depth,
        ParentInfo* parent_record,
        FrontierRecord* res
    ) except -1 nogil
