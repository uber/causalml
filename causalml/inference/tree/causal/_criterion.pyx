# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True

from libc.math cimport fabs
from libc.string cimport memset
from libc.string cimport memcpy


cdef class CausalRegressionCriterion(RegressionCriterion):
    """
    Base class for causal tree criterion
    """
    cdef public SplitState state
    cdef public float64_t groups_penalty

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const int32_t[:] treatment,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.treatment = treatment
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
        cdef int32_t is_treated
        cdef intp_t k = 0
        cdef float64_t w = 1.0

        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(float64_t))
        self.sq_sum_total = 0.
        self.state.node = [0., 0., 0., 0., 0., 0., 0., 0., 1.]
        self.state.left = [0., 0., 0., 0., 0., 0., 0., 0., 1.]
        self.state.right = [0., 0., 0., 0., 0., 0., 0., 0., 1.]

        for p in range(start, end):
            i = sample_indices[p]
            is_treated = treatment[i]

            self.sum_total[k] += self.y[i, k]
            self.sq_sum_total += self.y[i, k] * self.y[i, k]
            self.weighted_n_node_samples += w

            self.state.node.tr_count += is_treated
            self.state.node.tr_y_sum += is_treated * self.y[i, k]
            self.state.node.tr_y_sq_sum += is_treated * self.y[i, k] * self.y[i, k]

        self.state.node.ct_count = self.weighted_n_node_samples - self.state.node.tr_count
        self.state.node.ct_y_sum = self.sum_total[k] - self.state.node.tr_y_sum
        self.state.node.ct_y_sq_sum = self.sq_sum_total - self.state.node.tr_y_sq_sum

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef intp_t n_bytes = self.n_outputs * sizeof(float64_t)

        memset(&self.sum_left[0], 0, n_bytes)
        memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)

        self.state.left.y_sq_sum = 0.
        self.state.left.tr_y_sq_sum = 0.
        self.state.left.tr_y_sum = 0.
        self.state.left.ct_y_sq_sum = 0.
        self.state.left.ct_y_sum = 0.

        self.state.right.y_sq_sum = self.sq_sum_total
        self.state.right.tr_y_sq_sum = self.state.node.tr_y_sq_sum
        self.state.right.tr_y_sum = self.state.node.tr_y_sum
        self.state.right.ct_y_sq_sum = self.state.node.ct_y_sq_sum
        self.state.right.ct_y_sum = self.state.node.ct_y_sum

        self.state.left.count = 0.
        self.state.left.tr_count = 0.
        self.state.left.ct_count = 0.

        self.state.right.count = self.state.node.tr_count + self.state.node.ct_count
        self.state.right.tr_count = self.state.node.tr_count
        self.state.right.ct_count = self.state.node.ct_count

        # For compatibility with sklearn functions
        self.weighted_n_left = 0.
        self.weighted_n_right = self.weighted_n_node_samples

        self.pos = self.start

        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef intp_t n_bytes = self.n_outputs * sizeof(float64_t)
        memset(&self.sum_right[0], 0, n_bytes)
        memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)

        self.state.right.y_sq_sum = 0.
        self.state.right.tr_y_sq_sum = 0.
        self.state.right.tr_y_sum = 0.
        self.state.right.ct_y_sq_sum = 0.
        self.state.right.ct_y_sum = 0.

        self.state.left.y_sq_sum = self.sq_sum_total
        self.state.left.tr_y_sq_sum = self.state.node.tr_y_sq_sum
        self.state.left.tr_y_sum = self.state.node.tr_y_sum
        self.state.left.ct_y_sq_sum = self.state.node.ct_y_sq_sum
        self.state.left.ct_y_sum = self.state.node.ct_y_sum

        self.state.right.count = 0.
        self.state.right.tr_count = 0.
        self.state.right.ct_count = 0.

        self.state.left.count = self.state.node.tr_count + self.state.node.ct_count
        self.state.left.tr_count = self.state.node.tr_count
        self.state.left.ct_count = self.state.node.ct_count

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples

        self.pos = self.end

        return 0

    cdef int update(self, intp_t new_pos) nogil except -1:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const int32_t[:] treatment = self.treatment
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t pos = self.pos
        cdef intp_t end = self.end
        cdef intp_t i
        cdef int32_t is_treated
        cdef intp_t p
        cdef intp_t k = 0
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
                is_treated = treatment[i]

                self.sum_left[k] += self.y[i, k]
                self.state.left.tr_y_sum += is_treated * self.y[i, k]
                self.state.left.tr_y_sq_sum += is_treated * self.y[i, k] * self.y[i, k]
                self.state.left.ct_y_sum += (1. - is_treated) * self.y[i, k]
                self.state.left.ct_y_sq_sum += (1. - is_treated) * self.y[i, k] * self.y[i, k]
                self.state.left.tr_count += is_treated
                self.state.left.ct_count += (1. - is_treated)

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]
                is_treated = treatment[i]

                self.sum_left[k] -= self.y[i, k]
                self.state.left.tr_y_sum -= is_treated * self.y[i, k]
                self.state.left.tr_y_sq_sum -= is_treated * self.y[i, k] * self.y[i, k]
                self.state.left.ct_y_sum -= (1. - is_treated) * self.y[i, k]
                self.state.left.ct_y_sq_sum -= (1. - is_treated) * self.y[i, k] * self.y[i, k]
                self.state.left.tr_count -= is_treated
                self.state.left.ct_count -= (1. - is_treated)

                self.weighted_n_left -= w

        self.state.left.count = self.state.left.tr_count + self.state.left.ct_count
        self.state.right.tr_count = self.state.node.tr_count - self.state.left.tr_count
        self.state.right.ct_count = self.state.node.ct_count - self.state.left.ct_count
        self.state.right.count = self.state.right.tr_count + self.state.right.ct_count

        self.state.right.tr_y_sum = self.state.node.tr_y_sum - self.state.left.tr_y_sum
        self.state.right.ct_y_sum = self.state.node.ct_y_sum - self.state.left.ct_y_sum

        self.state.right.tr_y_sq_sum = self.state.node.tr_y_sq_sum - self.state.left.tr_y_sq_sum
        self.state.right.ct_y_sq_sum = self.state.node.ct_y_sq_sum - self.state.left.ct_y_sq_sum

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.sum_right[k] = self.sum_total[k] - self.sum_left[k]
        self.pos = new_pos

        return 0

    cdef void node_value(self, float64_t * dest) noexcept nogil:
        """Compute the node values of sample_indices[start:end] into dest."""
        dest[0] = self.state.node.ct_y_sum / self.state.node.ct_count
        dest[1] = self.state.node.tr_y_sum / self.state.node.tr_count

    cdef float64_t get_groups_penalty(self, float64_t tr_count, float64_t ct_count) nogil:
        """Compute penalty for the sample size difference between groups"""
        return self.groups_penalty * fabs(tr_count- ct_count)


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

        impurity += self.get_groups_penalty(self.state.node.tr_count, self.state.node.ct_count)

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

        penalty_left = self.get_groups_penalty(self.state.left.tr_count, self.state.left.ct_count)
        penalty_right = self.get_groups_penalty(self.state.right.tr_count, self.state.right.ct_count)

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
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] += self.get_groups_penalty(self.state.left.tr_count, self.state.left.ct_count)
        impurity_right[0] += self.get_groups_penalty(self.state.right.tr_count, self.state.right.ct_count)

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
        cdef float64_t impurity
        cdef float64_t node_tau
        cdef float64_t tr_var
        cdef float64_t ct_var

        node_tau = self.get_tau(self.state.node)
        tr_var = self.get_variance(
            self.state.node.tr_y_sum,
            self.state.node.tr_y_sq_sum,
            self.state.node.tr_count
        )
        ct_var = self.get_variance(
            self.state.node.ct_y_sum,
            self.state.node.ct_y_sq_sum,
            self.state.node.ct_count)
        impurity = (tr_var / self.state.node.tr_count + ct_var / self.state.node.ct_count) - node_tau * node_tau
        impurity += self.get_groups_penalty(self.state.node.tr_count, self.state.node.ct_count)

        return impurity

    cdef float64_t get_tau(self, NodeInfo info) nogil:
        return info.tr_y_sum / info.tr_count - info.ct_y_sum / info.ct_count

    cdef float64_t get_variance(self, float64_t y_sum, float64_t y_sq_sum, float64_t count) nogil:
        return  y_sq_sum / count - (y_sum * y_sum) / (count * count)

    cdef void children_impurity(self, float64_t * impurity_left, float64_t * impurity_right) noexcept nogil:
        """
        Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (sample_indices[start:pos]) and the impurity the right child
           (sample_indices[pos:end]).
        """

        cdef float64_t right_tr_var
        cdef float64_t right_ct_var
        cdef float64_t left_tr_var
        cdef float64_t left_ct_var

        cdef float64_t right_tau
        cdef float64_t left_tau

        right_tau = self.get_tau(self.state.right)
        right_tr_var = self.get_variance(
            self.state.right.tr_y_sum , self.state.right.tr_y_sq_sum, self.state.right.tr_count)
        right_ct_var = self.get_variance(
            self.state.right.ct_y_sum, self.state.right.ct_y_sq_sum, self.state.right.ct_count)

        left_tau = self.get_tau(self.state.left)
        left_tr_var = self.get_variance(
            self.state.left.tr_y_sum , self.state.left.tr_y_sq_sum, self.state.left.tr_count)
        left_ct_var = self.get_variance(
            self.state.left.ct_y_sum, self.state.left.ct_y_sq_sum, self.state.left.ct_count)

        impurity_left[0] = (left_tr_var / self.state.left.tr_count + left_ct_var / self.state.left.ct_count) - \
                           left_tau * left_tau
        impurity_right[0] = (right_tr_var / self.state.right.tr_count + right_ct_var / self.state.right.ct_count) - \
                            right_tau * right_tau

        impurity_left[0]  += self.get_groups_penalty(self.state.left.tr_count, self.state.left.ct_count)
        impurity_right[0] += self.get_groups_penalty(self.state.right.tr_count, self.state.right.ct_count)


cdef class TTest(CausalRegressionCriterion):
    """
    TTest impurity criterion for Causal Tree based on "Su, Xiaogang, et al. (2009). Subgroup analysis via recursive partitioning."
    """
    cdef float64_t node_impurity(self) noexcept nogil:
        cdef float64_t impurity
        cdef float64_t node_tau
        cdef float64_t tr_var
        cdef float64_t ct_var

        node_tau = self.get_tau(self.state.node)
        tr_var = self.get_variance(
            self.state.node.tr_y_sum,
            self.state.node.tr_y_sq_sum,
            self.state.node.tr_count
        )
        ct_var = self.get_variance(
            self.state.node.ct_y_sum,
            self.state.node.ct_y_sq_sum,
            self.state.node.ct_count)
        # T statistic of difference between treatment and control means
        impurity = node_tau / (((tr_var / self.state.node.tr_count) + (ct_var / self.state.node.ct_count)) ** 0.5)

        return impurity

    cdef float64_t get_tau(self, NodeInfo info) nogil:
        return info.tr_y_sum / info.tr_count - info.ct_y_sum / info.ct_count

    cdef float64_t get_variance(self, float64_t y_sum, float64_t y_sq_sum, float64_t count) nogil:
        return y_sq_sum / count - (y_sum * y_sum) / (count * count)

    cdef void children_impurity(self, float64_t * impurity_left, float64_t * impurity_right) noexcept nogil:
        """
        Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (sample_indices[start:pos]) and the impurity the right child
           (sample_indices[pos:end]).
        """
        cdef float64_t right_tr_var
        cdef float64_t right_ct_var
        cdef float64_t left_tr_var
        cdef float64_t left_ct_var
        cdef float64_t right_tau
        cdef float64_t left_tau
        cdef float64_t right_t_stat
        cdef float64_t left_t_stat
        cdef float64_t t_stat
        cdef float64_t pooled_var

        right_tau = self.get_tau(self.state.right)
        right_tr_var = self.get_variance(
            self.state.right.tr_y_sum,
            self.state.right.tr_y_sq_sum,
            self.state.right.tr_count)
        right_ct_var = self.get_variance(
            self.state.right.ct_y_sum,
            self.state.right.ct_y_sq_sum,
            self.state.right.ct_count)

        left_tau = self.get_tau(self.state.left)
        left_tr_var = self.get_variance(
            self.state.left.tr_y_sum,
            self.state.left.tr_y_sq_sum,
            self.state.left.tr_count)
        left_ct_var = self.get_variance(
            self.state.left.ct_y_sum,
            self.state.left.ct_y_sq_sum,
            self.state.left.ct_count)
        pooled_var = ((self.state.right.tr_count - 1) / (
                    self.state.node.tr_count + self.state.node.ct_count - 4)) * right_tr_var + \
                     (self.state.right.ct_count - 1) / (
                                 self.state.node.tr_count + self.state.node.ct_count - 4) * right_ct_var + \
                     (self.state.left.tr_count - 1) / (
                                 self.state.node.tr_count + self.state.node.ct_count - 4) * left_tr_var + \
                     (self.state.left.ct_count - 1) / (
                                 self.state.node.tr_count + self.state.node.ct_count - 4) * left_ct_var

        # T statistic of difference between treatment and control means in left and right nodes
        left_t_stat = left_tau / (
                    ((left_ct_var / self.state.left.ct_count) + (left_tr_var / self.state.left.tr_count)) ** 0.5)
        right_t_stat = right_tau / (
                    ((right_ct_var / self.state.right.ct_count) + (right_tr_var / self.state.right.tr_count)) ** 0.5)

        # Squared T statistic of difference between tau from left and right nodes.
        t_stat = ((left_tau - right_tau) / ((pooled_var ** 0.5) * (
                    (1 / self.state.right.tr_count) + (1 / self.state.right.ct_count) + (
                        1 / self.state.left.tr_count) + (1 / self.state.left.ct_count)) ** 0.5)) ** 2

        self.state.left.split_metric = t_stat+self.get_groups_penalty(self.state.node.tr_count,
                                                                      self.state.node.ct_count)

        impurity_left[0] = left_t_stat
        impurity_right[0] = right_t_stat

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
