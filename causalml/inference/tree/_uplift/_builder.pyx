# distutils: language = c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

"""Kernel tree builders for uplift trees with parent-shrinkage regularization.

These mirror the stock ``DepthFirstTreeBuilder`` / ``BestFirstTreeBuilder`` in
``_tree/_tree.pyx`` (same stopping rules, same ``__cinit__`` signatures) with two
uplift-specific changes:

1. **Parent-summary threading.** The regularization in
   :class:`UpliftClassificationCriterion` shrinks each node toward its parent's
   *regularized* ``P(Y=1|T=g)``. That summary is not derivable from the raw leaf
   values stored in ``tree.value`` (which stay raw for prediction parity with the
   legacy tree), so the builder keeps a ``node_id -> regularized summary`` map: it
   sets the parent summary on the criterion before each ``node_split`` and stores
   this node's regularized summary after ``node_value``, for its children to read.

2. **No divergence-based early-leaf.** The stock builders turn a node into a leaf
   when ``impurity <= EPSILON``. For uplift the "impurity" is a constant placeholder
   and growth is driven by the split gain; shrinkage further shrinks child
   divergences toward zero, so keeping that check would prematurely stop growth and
   diverge from the legacy tree (which leafs only on depth / min-samples / a
   non-positive best gain). The check is therefore omitted here.
"""

from libc.stdint cimport INTPTR_MAX
from libcpp cimport bool
from libcpp.stack cimport stack
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.algorithm cimport pop_heap
from libcpp.algorithm cimport push_heap

from ._criterion cimport UpliftClassificationCriterion

import numpy as np
cimport numpy as np
np.import_array()


cdef float64_t INFINITY = np.inf
cdef float64_t EPSILON = np.finfo('double').eps

cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED


cdef class DepthFirstUpliftTreeBuilder(TreeBuilder):
    """Build an uplift tree in depth-first fashion.

    DepthFirstTreeBuilder with uplift parent-summary threading.
    Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx
    """

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, object X,
                const float64_t[:, ::1] y,
                const float64_t[:] sample_weight=None,
                const unsigned char[::1] missing_values_in_feature_mask=None,
                ):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Initial capacity
        cdef intp_t init_capacity

        if tree.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        cdef intp_t start
        cdef intp_t end
        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef intp_t n_node_samples = splitter.n_samples
        cdef float64_t weighted_n_node_samples
        cdef SplitRecord split
        cdef intp_t node_id

        cdef float64_t middle_value
        cdef float64_t left_child_min
        cdef float64_t left_child_max
        cdef float64_t right_child_min
        cdef float64_t right_child_max
        cdef bint is_leaf
        cdef bint first = 1
        cdef intp_t max_depth_seen = -1
        cdef int rc = 0

        cdef stack[StackRecord] builder_stack
        cdef StackRecord stack_record

        cdef ParentInfo parent_record
        _init_parent_record(&parent_record)

        # Uplift regularization: parent regularized summary threading.
        cdef map[intp_t, vector[float64_t]] reg_map
        cdef vector[float64_t] node_summary
        node_summary.resize(tree.n_outputs, 0.0)

        with nogil:
            # push root node onto stack
            builder_stack.push({
                "start": 0,
                "end": n_node_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "impurity": INFINITY,
                "n_constant_features": 0,
                "lower_bound": -INFINITY,
                "upper_bound": INFINITY,
            })

            while not builder_stack.empty():
                stack_record = builder_stack.top()
                builder_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                parent_record.impurity = stack_record.impurity
                parent_record.n_constant_features = stack_record.n_constant_features
                parent_record.lower_bound = stack_record.lower_bound
                parent_record.upper_bound = stack_record.upper_bound

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                # Thread the parent's regularized summary into the criterion before
                # the split search (the root has none).
                if parent == _TREE_UNDEFINED:
                    (<UpliftClassificationCriterion> splitter.criterion).set_parent_summary(NULL, 0)
                else:
                    (<UpliftClassificationCriterion> splitter.criterion).set_parent_summary(&reg_map[parent][0], 1)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    parent_record.impurity = splitter.node_impurity()
                    first = 0

                if not is_leaf:
                    splitter.node_split(
                        &parent_record,
                        &split,
                    )
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, parent_record.impurity,
                                         n_node_samples, weighted_n_node_samples,
                                         split.missing_go_to_left)

                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)
                if splitter.with_monotonic_cst:
                    splitter.clip_node_value(tree.value + node_id * tree.value_stride, parent_record.lower_bound, parent_record.upper_bound)

                # Record this node's regularized summary for its children to shrink toward.
                (<UpliftClassificationCriterion> splitter.criterion).compute_reg_summary(&node_summary[0])
                reg_map[node_id] = node_summary

                if not is_leaf:
                    if (
                        not splitter.with_monotonic_cst or
                        splitter.monotonic_cst[split.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = parent_record.lower_bound
                        left_child_max = right_child_max = parent_record.upper_bound
                    elif splitter.monotonic_cst[split.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = parent_record.lower_bound
                        right_child_max = parent_record.upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        right_child_min = middle_value
                        left_child_max = middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = parent_record.lower_bound
                        left_child_max = parent_record.upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        middle_value = splitter.criterion.middle_value()
                        left_child_min = middle_value
                        right_child_max = middle_value

                    # Push right child on stack
                    builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": right_child_min,
                        "upper_bound": right_child_max,
                    })

                    # Push left child on stack
                    builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": left_child_min,
                        "upper_bound": left_child_max,
                    })

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


cdef inline bool _compare_records(
    const FrontierRecord& left,
    const FrontierRecord& right,
):
    return left.improvement < right.improvement


cdef inline void _add_to_frontier(
    FrontierRecord rec,
    vector[FrontierRecord]& frontier,
) noexcept nogil:
    """Adds record `rec` to the priority queue `frontier`."""
    frontier.push_back(rec)
    push_heap(frontier.begin(), frontier.end(), &_compare_records)


cdef class BestFirstUpliftTreeBuilder(TreeBuilder):
    """Build an uplift tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest split gain. BestFirstTreeBuilder with uplift parent-summary threading.
    Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx
    """
    cdef intp_t max_leaf_nodes
    # node_id -> regularized summary, threaded to each node's children. Instance
    # state because it must survive across ``_add_split_node`` calls.
    cdef map[intp_t, vector[float64_t]] reg_map

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf,  min_weight_leaf,
                  intp_t max_depth, intp_t max_leaf_nodes,
                  float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef intp_t max_leaf_nodes = self.max_leaf_nodes

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        cdef vector[FrontierRecord] frontier
        cdef FrontierRecord record
        cdef FrontierRecord split_node_left
        cdef FrontierRecord split_node_right
        cdef float64_t left_child_min
        cdef float64_t left_child_max
        cdef float64_t right_child_min
        cdef float64_t right_child_max

        cdef intp_t n_node_samples = splitter.n_samples
        cdef intp_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef intp_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        cdef ParentInfo parent_record
        _init_parent_record(&parent_record)

        self.reg_map.clear()

        # Initial capacity
        cdef intp_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = self._add_split_node(
                splitter=splitter,
                tree=tree,
                start=0,
                end=n_node_samples,
                is_first=IS_FIRST,
                is_left=IS_LEFT,
                parent=NULL,
                depth=0,
                parent_record=&parent_record,
                res=&split_node_left,
            )
            if rc >= 0:
                _add_to_frontier(split_node_left, frontier)

            while not frontier.empty():
                pop_heap(frontier.begin(), frontier.end(), &_compare_records)
                record = frontier.back()
                frontier.pop_back()

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    if (
                        not splitter.with_monotonic_cst or
                        splitter.monotonic_cst[node.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = record.lower_bound
                        left_child_max = right_child_max = record.upper_bound
                    elif splitter.monotonic_cst[node.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = record.lower_bound
                        right_child_max = record.upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        right_child_min = record.middle_value
                        left_child_max = record.middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = record.lower_bound
                        left_child_max = record.upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        left_child_min = record.middle_value
                        right_child_max = record.middle_value

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    parent_record.lower_bound = left_child_min
                    parent_record.upper_bound = left_child_max
                    parent_record.impurity = record.impurity_left
                    rc = self._add_split_node(
                        splitter=splitter,
                        tree=tree,
                        start=record.start,
                        end=record.pos,
                        is_first=IS_NOT_FIRST,
                        is_left=IS_LEFT,
                        parent=node,
                        depth=record.depth + 1,
                        parent_record=&parent_record,
                        res=&split_node_left,
                    )
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    parent_record.lower_bound = right_child_min
                    parent_record.upper_bound = right_child_max
                    parent_record.impurity = record.impurity_right
                    rc = self._add_split_node(
                        splitter=splitter,
                        tree=tree,
                        start=record.pos,
                        end=record.end,
                        is_first=IS_NOT_FIRST,
                        is_left=IS_NOT_LEFT,
                        parent=node,
                        depth=record.depth + 1,
                        parent_record=&parent_record,
                        res=&split_node_right,
                    )
                    if rc == -1:
                        break

                    # Add nodes to queue
                    _add_to_frontier(split_node_left, frontier)
                    _add_to_frontier(split_node_right, frontier)

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(
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
    ) except -1 nogil:
        """Adds node w/ partition ``[start, end)`` to the frontier. """
        cdef SplitRecord split
        cdef intp_t node_id
        cdef intp_t n_node_samples
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease
        cdef float64_t weighted_n_node_samples
        cdef bint is_leaf

        cdef vector[float64_t] node_summary

        splitter.node_reset(start, end, &weighted_n_node_samples)

        # Thread the parent's regularized summary into the criterion before the
        # split search (the root has none).
        if parent == NULL:
            (<UpliftClassificationCriterion> splitter.criterion).set_parent_summary(NULL, 0)
        else:
            (<UpliftClassificationCriterion> splitter.criterion).set_parent_summary(&self.reg_map[parent - tree.nodes][0], 1)

        # reset n_constant_features for this specific split before beginning split search
        parent_record.n_constant_features = 0

        if is_first:
            parent_record.impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (depth >= self.max_depth or
                   n_node_samples < self.min_samples_split or
                   n_node_samples < 2 * self.min_samples_leaf or
                   weighted_n_node_samples < 2 * self.min_weight_leaf
                   )

        if not is_leaf:
            splitter.node_split(
                parent_record,
                &split,
            )
            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                       split.improvement + EPSILON < min_impurity_decrease)

        node_id = tree._add_node(parent - tree.nodes
                                 if parent != NULL
                                 else _TREE_UNDEFINED,
                                 is_left, is_leaf,
                                 split.feature, split.threshold, parent_record.impurity,
                                 n_node_samples, weighted_n_node_samples,
                                 split.missing_go_to_left)
        if node_id == INTPTR_MAX:
            return -1

        # compute values also for split nodes (might become leafs later).
        splitter.node_value(tree.value + node_id * tree.value_stride)
        if splitter.with_monotonic_cst:
            splitter.clip_node_value(tree.value + node_id * tree.value_stride, parent_record.lower_bound, parent_record.upper_bound)

        # Record this node's regularized summary for its children to shrink toward.
        node_summary.resize(tree.n_outputs, 0.0)
        (<UpliftClassificationCriterion> splitter.criterion).compute_reg_summary(&node_summary[0])
        self.reg_map[node_id] = node_summary

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = parent_record.impurity
        res.lower_bound = parent_record.lower_bound
        res.upper_bound = parent_record.upper_bound
        res.middle_value = splitter.criterion.middle_value()

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = parent_record.impurity
            res.impurity_right = parent_record.impurity

        return 0
