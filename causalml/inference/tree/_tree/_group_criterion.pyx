# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# distutils: language = c++


cdef int32_t CONTROL_GROUP_IDX = 0


cdef class NodeState:

    def __cinit__(self):
        self.split_metric = 1.
        self.control_idx = CONTROL_GROUP_IDX
        self.control_total = 0
        self.treatment_total = 0
        self.groups_total = 0

    cdef int32_t reset(self, intp_t n_outputs) except -1 nogil:

        if self.count_1d.size() == 0:
            self.count_1d.resize(n_outputs, 0.)
            self.y_sum_1d.resize(n_outputs, 0.)
            self.y_sq_sum_1d.resize(n_outputs, 0.)
        else:
            self.count_1d.assign(n_outputs, 0.)
            self.y_sum_1d.assign(n_outputs, 0.)
            self.y_sq_sum_1d.assign(n_outputs, 0.)

        self.update_counters()
        return 0

    cdef int32_t update_counters(self) except -1 nogil:

        cdef int n_outputs = self.count_1d.size()

        if n_outputs == 0:
            return -1

        self.groups_total = n_outputs
        self.control_total = <int32_t> self.count_1d[self.control_idx]
        self.treatment_total = 0
        for k in range(n_outputs):
            if k != self.control_idx:
                self.treatment_total += <int32_t> self.count_1d[k]
        return 0

    cdef int32_t copy_from_state(self, NodeState state) except -1 nogil:

        if self.count_1d.size() == 0:
            return -1

        for k in range(self.count_1d.size()):
            self.count_1d[k] = state.count_1d[k]
            self.y_sum_1d[k] = state.y_sum_1d[k]
            self.y_sq_sum_1d[k] = state.y_sq_sum_1d[k]
        self.update_counters()
        return 0

    cdef int32_t set_difference(self, NodeState a, NodeState b) except -1 nogil:
        """Set this state to the per-group difference ``a - b``."""
        if self.count_1d.size() == 0:
            return -1

        for k in range(self.count_1d.size()):
            self.count_1d[k] = a.count_1d[k] - b.count_1d[k]
            self.y_sum_1d[k] = a.y_sum_1d[k] - b.y_sum_1d[k]
            self.y_sq_sum_1d[k] = a.y_sq_sum_1d[k] - b.y_sq_sum_1d[k]
        self.update_counters()
        return 0

    cdef int32_t increment_count(self, int32_t group_idx, float64_t value) except -1 nogil:
        self.count_1d[group_idx] += value
        self.update_counters()
        return 0

    cdef int32_t increment_y_sum(self, int32_t group_idx, float64_t value) except -1 nogil:
        self.y_sum_1d[group_idx] += value
        return 0

    cdef int32_t increment_y_sq_sum(self, int32_t group_idx, float64_t value) except -1 nogil:
        self.y_sq_sum_1d[group_idx] += value
        return 0

    cdef float64_t outcome_mean(self, int32_t group_idx) noexcept nogil:
        return self.y_sum_1d[group_idx] / self.count_1d[group_idx]

    cdef float64_t outcome_var(self, int32_t group_idx) noexcept nogil:
        cdef float64_t var
        var = (self.y_sq_sum_1d[group_idx] / self.count_1d[group_idx] -
                (self.y_sum_1d[group_idx] * self.y_sum_1d[group_idx]) / (
                            self.count_1d[group_idx] * self.count_1d[group_idx]))
        # Clamp tiny negative variance to 0 instead of returning -1
        var = max(var, 0.0)
        return var

    cdef float64_t effect(self, int32_t treatment_idx) noexcept nogil:
        return (self.y_sum_1d[treatment_idx] / self.count_1d[treatment_idx] -
                self.y_sum_1d[self.control_idx] / self.count_1d[self.control_idx])


cdef class NodeSplitState:

    def __cinit__(self, intp_t n_outputs):
        self.node = NodeState(n_outputs)
        self.right = NodeState(n_outputs)
        self.left = NodeState(n_outputs)
        self.missing = NodeState(n_outputs)
        self.reset_nodes(n_outputs)

    cdef int32_t reset_nodes(self, intp_t n_outputs) except -1 nogil:
        self.node.reset(n_outputs)
        self.right.reset(n_outputs)
        self.left.reset(n_outputs)
        self.missing.reset(n_outputs)
        return 0


cdef class CausalRegressionCriterion(RegressionCriterion):
    """
    Base class for causal tree criterion
    """

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        # Parent __cinit__ is automatically called
        self.state = NodeSplitState(n_outputs)

    cdef int get_group_stats(
        self,
        int32_t* groups_count,
        int64_t* tr_count_mean,
        int32_t* ct_count,
        int32_t* min_size_among_groups,
        ) except -1 nogil:

        cdef int32_t min_size = <int32_t> self.state.node.count_1d[0]
        for k in range(1, self.n_outputs):
            min_size = <int32_t>  self.state.node.count_1d[k] if <int32_t> self.state.node.count_1d[k]  < min_size else min_size
        cdef int32_t groups = <int32_t> self.state.node.groups_total

        min_size_among_groups[0] = min_size
        groups_count[0] = groups
        ct_count[0] = <int32_t> self.state.node.count_1d[self.state.node.control_idx]
        tr_count_mean[0] = <int64_t> ( (<int64_t> self.state.node.treatment_total) / (<int64_t> (groups - 1)) )

        return 0

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Initialize the criterion.
        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Notes:
        1) self.y[i, k] is nan if a particular observation is not in a group k, k is in range(0, n_outputs - 1).
        2) Control group index is fixed to 0 value.
        3) Impurity is averaged across the impurity vector calculated for all pairs of 
           control & treatment_i, i is in range(1, n_outputs - 1)
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        # For compatibility with sklearn functions
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0
        cdef float64_t y_ik
        cdef float64_t w_y_ik

        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        self.sq_sum_total = 0.
        self.state.reset_nodes(self.n_outputs)

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            # k is the number of groups
            for k in range(self.n_outputs):
                y_ik = self.y[i, k]

                if not isnan(y_ik):
                    w_y_ik = w * y_ik
                    self.sum_total[k] += w_y_ik
                    self.sq_sum_total += w_y_ik * y_ik
                    self.weighted_n_node_samples += w

                    # Add groups statistics into node state
                    self.state.node.increment_count(k, 1.)
                    self.state.node.increment_y_sum(k, w_y_ik)
                    self.state.node.increment_y_sq_sum(k, w_y_ik * y_ik)

        # Reset to pos=start
        self.reset()
        return 0

    # ``init_sum_missing`` is inherited from ``RegressionCriterion`` (it only
    # allocates the flat ``sum_missing`` buffer); the per-group ``state.missing``
    # is created in ``NodeSplitState.__cinit__``.
    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Accumulate the missing samples' statistics for the current feature.

        Mirrors ``RegressionCriterion.init_missing`` (scikit-learn): the splitter
        places the ``n_missing`` samples that have a missing value for the current
        feature at ``sample_indices[end - n_missing:end]``. Their contribution is
        summed both into the flat ``sum_missing`` (used by ``StandardMSE``) and the
        per-group ``state.missing`` (used by the per-group causal / uplift
        criteria), still respecting the group-encoding NaN mask ``isnan(y[i, k])``
        (``y[i, k]`` is NaN when sample ``i`` is not in group ``k``).
        """
        cdef intp_t i, p, k
        cdef float64_t w = 1.0
        cdef float64_t y_ik, w_y_ik

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.sum_missing[0], 0, self.n_outputs * sizeof(float64_t))
        self.weighted_n_missing = 0.0
        self.state.missing.reset(self.n_outputs)

        # The missing samples are assumed to be in sample_indices[end - n_missing:end].
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]

            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                if not isnan(y_ik):
                    w_y_ik = w * y_ik
                    self.sum_missing[k] += w_y_ik
                    self.state.missing.increment_count(k, 1.)
                    self.state.missing.increment_y_sum(k, w_y_ik)
                    self.state.missing.increment_y_sq_sum(k, w_y_ik * y_ik)
                    self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        cdef intp_t n_bytes = self.n_outputs * sizeof(float64_t)
        cdef intp_t k

        if self.n_missing != 0 and self.missing_go_to_left:
            # Missing samples seed the left child (scikit-learn missing_go_to_left).
            memcpy(&self.sum_left[0], &self.sum_missing[0], n_bytes)
            for k in range(self.n_outputs):
                self.sum_right[k] = self.sum_total[k] - self.sum_missing[k]

            self.state.left.copy_from_state(self.state.missing)
            self.state.right.set_difference(self.state.node, self.state.missing)

            self.weighted_n_left = self.weighted_n_missing
            self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_missing
        else:
            memset(&self.sum_left[0], 0, n_bytes)
            memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)

            self.state.left.reset(self.n_outputs)
            self.state.right.copy_from_state(self.state.node)

            # For compatibility with sklearn functions
            self.weighted_n_left = 0.
            self.weighted_n_right = self.weighted_n_node_samples

        self.pos = self.start

        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        cdef intp_t n_bytes = self.n_outputs * sizeof(float64_t)
        cdef intp_t k

        if self.n_missing != 0 and not self.missing_go_to_left:
            # Missing samples seed the right child.
            memcpy(&self.sum_right[0], &self.sum_missing[0], n_bytes)
            for k in range(self.n_outputs):
                self.sum_left[k] = self.sum_total[k] - self.sum_missing[k]

            self.state.right.copy_from_state(self.state.missing)
            self.state.left.set_difference(self.state.node, self.state.missing)

            self.weighted_n_right = self.weighted_n_missing
            self.weighted_n_left = self.weighted_n_node_samples - self.weighted_n_missing
        else:
            memset(&self.sum_right[0], 0, n_bytes)
            memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)

            self.state.right.reset(self.n_outputs)
            self.state.left.copy_from_state(self.state.node)

            # For compatibility with sklearn functions
            self.weighted_n_right = 0.0
            self.weighted_n_left = self.weighted_n_node_samples

        self.pos = self.end

        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t pos = self.pos
        # Missing samples are held at sample_indices[end - n_missing:end] and are
        # assigned to a fixed child by reset()/reverse_reset(); the scan only ever
        # moves the non-missing samples (scikit-learn convention).
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef intp_t i
        cdef intp_t p
        cdef intp_t k = 0
        cdef float64_t y_ik
        cdef float64_t w_y_ik
        cdef float64_t w = 1.0

        """
        Update statistics up to new_pos

        Given that:
            sum_total[x] = sum_left[x] + sum_right[x]
        we are going to update sum_left from the direction that require the least amount of computations,
        i.e. from pos to new_pos or from end to new_pos
        """
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    if not isnan(y_ik):
                        w_y_ik = w * y_ik
                        self.sum_left[k] += w_y_ik
                        self.state.left.increment_count(k, 1.)
                        self.state.left.increment_y_sum(k, w_y_ik)
                        self.state.left.increment_y_sq_sum(k, w_y_ik * y_ik)

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    if not isnan(y_ik):
                        w_y_ik = w * y_ik
                        self.sum_left[k] -= w_y_ik
                        self.state.left.increment_count(k, -1.)
                        self.state.left.increment_y_sum(k, -w_y_ik)
                        self.state.left.increment_y_sq_sum(k, -w_y_ik * y_ik)

                self.weighted_n_left -= w

        for k in range(self.n_outputs):
            self.state.right.count_1d[k] = self.state.node.count_1d[k] - self.state.left.count_1d[k]
            self.state.right.y_sum_1d[k] = self.state.node.y_sum_1d[k] - self.state.left.y_sum_1d[k]
            self.state.right.y_sq_sum_1d[k] = self.state.node.y_sq_sum_1d[k] - self.state.left.y_sq_sum_1d[k]

            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = new_pos

        return 0

    cdef void node_value(self, float64_t * dest) noexcept nogil:
        """Compute the node values of sample_indices[start:end] into dest."""
        cdef intp_t k
        for k in range(self.n_outputs):
            dest[k] = self.state.node.outcome_mean(k)

    cdef float64_t get_groups_penalty(self, NodeState node) noexcept nogil:
        """Compute penalty for sample size differences across multiple treatment groups.
        Penalizes imbalance of average absolute difference.
        """
        cdef intp_t k
        cdef int32_t groups_total = self.n_outputs
        cdef int32_t num_treatments = groups_total - 1
        cdef float64_t fabs_diff_sum = 0.0

        if num_treatments <= 0:
            return 0.0

        for k in range(groups_total):
            if k == node.control_idx:
                continue
            fabs_diff_sum += fabs(node.count_1d[k] - node.count_1d[CONTROL_GROUP_IDX])

        return self.groups_penalty * (fabs_diff_sum / <float64_t> num_treatments)



