# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True


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
        self.reset_nodes(n_outputs)

    cdef int32_t reset_nodes(self, intp_t n_outputs) except -1 nogil:
        self.node.reset(n_outputs)
        self.right.reset(n_outputs)
        self.left.reset(n_outputs)
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

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        cdef intp_t n_bytes = self.n_outputs * sizeof(float64_t)

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
        cdef intp_t end = self.end
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
        if (new_pos - pos) <= (end - new_pos):
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

            for p in range(end - 1, new_pos - 1, -1):
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



cdef class StandardMSE(CausalRegressionCriterion):
    """
    Standard MSE with treatment effect estimates
    Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx
    """

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef float64_t impurity
        cdef intp_t k


        impurity = self.sq_sum_total / self.n_node_samples
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.n_node_samples) ** 2.0

        impurity += self.get_groups_penalty(self.state.node)

        return impurity / self.n_outputs

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        The MSE proxy is derived from
            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
        Neglecting constant terms, this gives:
            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        cdef intp_t k
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0
        cdef float64_t penalty_left, penalty_right

        penalty_left = self.get_groups_penalty(self.state.left)
        penalty_right = self.get_groups_penalty(self.state.right)

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k] - penalty_left
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k] - penalty_right

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(
        self,
        float64_t * impurity_left,
        float64_t * impurity_right
    ) noexcept nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t pos = self.pos
        cdef intp_t start = self.start

        cdef float64_t y_ik

        cdef float64_t sq_sum_left = 0.0
        cdef float64_t sq_sum_right

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0

        cdef float64_t penalty_left, penalty_right

        for p in range(start, pos):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                if not isnan(y_ik):
                    sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] += self.get_groups_penalty(self.state.left)
        impurity_right[0] += self.get_groups_penalty(self.state.right)

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


cdef class CausalMSE(CausalRegressionCriterion):
    """
    Mean squared error impurity criterion for Causal Tree
    CausalTreeMSE = right_effect + left_effect
    where,
    effect = alpha * tau^2 - (1 - alpha) * (1 + train_to_est_ratio) * (VAR_tr / p + VAR_cont / (1 - p))
    """

    cdef float64_t node_impurity(self) noexcept nogil:
        """
        Evaluate the impurity of the current node, i.e. the impurity of sample_indices[start:end].
        """

        cdef float64_t impurity = 0.
        cdef int32_t tr_group_idx
        cdef float64_t node_tau
        cdef float64_t tr_var
        cdef float64_t ct_var = self.state.node.outcome_var(CONTROL_GROUP_IDX)
        cdef float64_t tr_count
        cdef float64_t ct_count = self.state.node.count_1d[CONTROL_GROUP_IDX]

        for tr_group_idx in range(1, self.n_outputs):
            node_tau = self.state.node.effect(tr_group_idx)
            tr_var = self.state.node.outcome_var(tr_group_idx)
            tr_count = self.state.node.count_1d[tr_group_idx]

            impurity += (tr_var / tr_count + ct_var / ct_count) - node_tau * node_tau

        impurity /= (self.n_outputs - 1)
        impurity += self.get_groups_penalty(self.state.node)

        return impurity

    cdef void children_impurity(self, float64_t * impurity_left, float64_t * impurity_right) noexcept nogil:
        """
        Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (sample_indices[start:pos]) and the impurity the right child
           (sample_indices[pos:end]).
        """

        cdef float64_t right_tr_var
        cdef float64_t right_ct_var = self.state.right.outcome_var(CONTROL_GROUP_IDX)
        cdef float64_t right_tr_count
        cdef float64_t right_ct_count = self.state.right.count_1d[CONTROL_GROUP_IDX]
        cdef float64_t left_tr_var
        cdef float64_t left_ct_var = self.state.left.outcome_var(CONTROL_GROUP_IDX)
        cdef float64_t left_tr_count
        cdef float64_t left_ct_count = self.state.left.count_1d[CONTROL_GROUP_IDX]
        cdef float64_t right_tau
        cdef float64_t left_tau

        impurity_right[0] = 0.
        impurity_left[0] = 0.

        for tr_group_idx in range(1, self.n_outputs):
            right_tau = self.state.right.effect(tr_group_idx)
            right_tr_var = self.state.right.outcome_var(tr_group_idx)
            right_tr_count = self.state.right.count_1d[tr_group_idx]

            left_tau = self.state.left.effect(tr_group_idx)
            left_tr_var = self.state.left.outcome_var(tr_group_idx)
            left_tr_count = self.state.left.count_1d[tr_group_idx]

            impurity_right[0] += (right_tr_var / right_tr_count + right_ct_var / right_ct_count) - right_tau * right_tau
            impurity_left[0] += (left_tr_var / left_tr_count + left_ct_var / left_ct_count) - left_tau * left_tau

        impurity_right[0] /= (self.n_outputs - 1)
        impurity_left[0] /= (self.n_outputs - 1)
        impurity_right[0] += self.get_groups_penalty(self.state.right)
        impurity_left[0] += self.get_groups_penalty(self.state.left)


cdef class TTest(CausalRegressionCriterion):
    """
    TTest impurity criterion for Causal Tree based on "Su, Xiaogang, et al. (2009). Subgroup analysis via recursive partitioning."
    """
    cdef float64_t node_impurity(self) noexcept nogil:
        

        cdef float64_t impurity = 0.
        cdef int32_t tr_group_idx
        cdef float64_t node_tau
        cdef float64_t tr_var
        cdef float64_t ct_var = self.state.node.outcome_var(CONTROL_GROUP_IDX)
        cdef float64_t tr_count
        cdef float64_t ct_count = self.state.node.count_1d[CONTROL_GROUP_IDX]
        cdef float64_t denom

        for tr_group_idx in range(1, self.n_outputs):
            node_tau = self.state.node.effect(tr_group_idx)
            tr_var = self.state.node.outcome_var(tr_group_idx)
            tr_count = self.state.node.count_1d[tr_group_idx]
            # T statistic of difference between treatment and control means
            denom = sqrt(( (tr_var / tr_count) + (ct_var / ct_count)))
            if denom > 0:
                impurity += node_tau / denom

        return impurity

    cdef void children_impurity(self, float64_t * impurity_left, float64_t * impurity_right) noexcept nogil:
        """
        Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (sample_indices[start:pos]) and the impurity the right child
           (sample_indices[pos:end]).
        """

        cdef int32_t tr_group_idx
        cdef int32_t num_treatments = self.n_outputs - 1

        cdef float64_t t_left_sum = 0.0
        cdef float64_t t_right_sum = 0.0
        cdef float64_t tdiff = 0.0
        cdef float64_t tdiff_sq_sum = 0.0

        cdef float64_t left_tau, right_tau
        cdef float64_t left_tr_var, right_tr_var
        cdef float64_t left_ct_var = self.state.left.outcome_var(CONTROL_GROUP_IDX)
        cdef float64_t right_ct_var = self.state.right.outcome_var(CONTROL_GROUP_IDX)

        cdef float64_t left_tr_count, right_tr_count
        cdef float64_t left_ct_count = self.state.left.count_1d[CONTROL_GROUP_IDX]
        cdef float64_t right_ct_count = self.state.right.count_1d[CONTROL_GROUP_IDX]

        cdef float64_t denom_left, denom_right
        cdef float64_t pooled_var_t
        cdef float64_t inv_n_sum
        cdef float64_t dof

        impurity_left[0] = 0.0
        impurity_right[0] = 0.0

        for tr_group_idx in range(1, self.n_outputs):
            right_tau = self.state.right.effect(tr_group_idx)
            right_tr_var = self.state.right.outcome_var(tr_group_idx)
            right_tr_count = self.state.right.count_1d[tr_group_idx]

            left_tau = self.state.left.effect(tr_group_idx)
            left_tr_var = self.state.left.outcome_var(tr_group_idx)
            left_tr_count = self.state.left.count_1d[tr_group_idx]

            denom_left = sqrt(left_tr_var / left_tr_count + left_ct_var / left_ct_count)
            denom_right = sqrt(right_tr_var / right_tr_count + right_ct_var / right_ct_count)
            if denom_left > 0.:
                t_left_sum += left_tau / denom_left
            if denom_right > 0.:
                t_right_sum += right_tau / denom_right
    
            # Per-treatment squared difference in taus between sides
            inv_n_sum = (1.0 / right_tr_count + 1.0 / right_ct_count +
                        1.0 / left_tr_count + 1.0 / left_ct_count)

            # Pooled variance across four cells (left/right × tr/ct)
            pooled_var_t = 0.0
            pooled_var_t += ((right_tr_count - 1.0) * right_tr_var)
            pooled_var_t += ((right_ct_count - 1.0) * right_ct_var)
            pooled_var_t += ((left_tr_count - 1.0) * left_tr_var)
            pooled_var_t += ((left_ct_count - 1.0) * left_ct_var)

            # Normalize by total degrees of freedom if it is positive
            dof = (right_tr_count - 1.0) + (right_ct_count - 1.0) + (left_tr_count - 1.0) + (left_ct_count - 1.0)
            if dof > 0.0:
                pooled_var_t /= dof

            if pooled_var_t > 0.0 and inv_n_sum > 0.0:
                tdiff = ((left_tau - right_tau) / (( sqrt(pooled_var_t) ) * ( sqrt(inv_n_sum) )))
                tdiff_sq_sum += (tdiff * tdiff)

        self.state.left.split_metric = (tdiff_sq_sum / <float64_t> num_treatments) + self.get_groups_penalty(self.state.node)
        
        impurity_left[0] = t_left_sum / <float64_t> num_treatments
        impurity_right[0] = t_right_sum / <float64_t> num_treatments

    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                     float64_t impurity_left,
                                     float64_t impurity_right) noexcept nogil:
        return self.state.left.split_metric

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction. In case of t statistic - proxy_impurity_improvement
        is the same as impurity_improvement.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return self.state.left.split_metric
