# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3


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

        # Missing samples (held at the tail) contribute their squared outcome to
        # the left child when they were routed left (scikit-learn convention); the
        # first-moment sums already carry them via reset()/update().
        if self.n_missing != 0 and self.missing_go_to_left:
            for p in range(self.end - self.n_missing, self.end):
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

            if tr_count > 0 and ct_count > 0:
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

            if right_tr_count > 0 and right_ct_count > 0:
                impurity_right[0] += (right_tr_var / right_tr_count + right_ct_var / right_ct_count) - right_tau * right_tau
            if left_tr_count > 0 and left_ct_count > 0:
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

            denom_left = 0.0
            denom_right = 0.0
            if left_tr_count > 0 and left_ct_count > 0:
                denom_left = sqrt(left_tr_var / left_tr_count + left_ct_var / left_ct_count)
            if right_tr_count > 0 and right_ct_count > 0:
                denom_right = sqrt(right_tr_var / right_tr_count + right_ct_var / right_ct_count)
            if denom_left > 0.:
                t_left_sum += left_tau / denom_left
            if denom_right > 0.:
                t_right_sum += right_tau / denom_right

            # Per-treatment squared difference in taus between sides
            inv_n_sum = 0.0
            if right_tr_count > 0 and right_ct_count > 0 and left_tr_count > 0 and left_ct_count > 0:
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
