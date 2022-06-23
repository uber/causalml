# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: linetrace=True

from sklearn.tree._criterion cimport RegressionCriterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t


cdef class CausalRegressionCriterion(RegressionCriterion):
    """
    Base class for causal tree criterion
    """
    cdef void node_value(self, double * dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t is_treated
        cdef DOUBLE_t y_ik

        cdef SIZE_t * samples = self.samples
        cdef DOUBLE_t * sample_weight = self.sample_weight

        cdef double node_ct = 0.0
        cdef double node_tr = 0.0
        cdef double node_ct_sum = 0.0
        cdef double node_tr_sum = 0.0
        cdef double eps = 1e-5

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                # the weights of 1 and 1 + eps are used for treatment and control respectively
                is_treated = sample_weight[i] - eps

            # assume that there is only one output (k = 0)
            y_ik = self.y[i, 0]

            node_tr += is_treated
            node_ct += 1. - is_treated
            node_tr_sum += y_ik * is_treated
            node_ct_sum += y_ik * (1. - is_treated)

        # save the average of treatment effects within a node as a value for the node
        dest[0] = node_tr_sum / node_tr - node_ct_sum / node_ct

cdef class StandardMSE(CausalRegressionCriterion):
    """
    Standard MSE with treatment effect estimates
    Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.weighted_n_node_samples) ** 2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
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
        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double * impurity_left,
                                double * impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef DOUBLE_t * sample_weight = self.sample_weight
        cdef SIZE_t * samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
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

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs

cdef class CausalMSE(CausalRegressionCriterion):
    """
    Mean squared error impurity criterion for Causal Tree
    CausalTreeMSE = right_effect + left_effect
    where,
    effect = alpha * tau^2 - (1 - alpha) * (1 + train_to_est_ratio) * (VAR_tr / p + VAR_cont / (1 - p))
    """

    cdef double node_impurity(self) nogil:
        """
        Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
        """

        cdef double * sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t is_treated
        cdef DOUBLE_t y_ik

        cdef SIZE_t * samples = self.samples
        cdef DOUBLE_t * sample_weight = self.sample_weight

        cdef double node_tr = 0.0
        cdef double node_ct = 0.0
        cdef double node_sum = self.sum_total[0]
        cdef double node_tr_sum = 0.0
        cdef double node_sq_sum = 0.0
        cdef double node_tr_sq_sum = 0.0
        cdef double tr_var
        cdef double ct_var
        cdef double eps = 1e-5

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                # It is enough to add eps to get zero values for control
                # treatment: 1 + eps, control: eps
                is_treated = sample_weight[i] - eps

            # assume that there is only one output (k = 0)
            y_ik = self.y[i, 0]

            node_tr += is_treated
            node_ct += (1. - is_treated)
            node_tr_sum += y_ik * is_treated
            node_sq_sum += y_ik * y_ik
            node_tr_sq_sum += y_ik * y_ik * is_treated

        # The average causal effect
        node_tau = node_tr_sum / node_tr - (node_sum - node_tr_sum) / node_ct
        # Outcome variance for treated
        tr_var = node_tr_sq_sum / node_tr - node_tr_sum * node_tr_sum / (node_tr * node_tr)
        # Outcome variance for control
        ct_var = ((node_sq_sum - node_tr_sq_sum) / node_ct -
                  (node_sum - node_tr_sum) * (node_sum - node_tr_sum) / (node_ct * node_ct))

        impurity = (tr_var / node_tr + ct_var / node_ct) - node_tau * node_tau

        return impurity

    cdef void children_impurity(self, double * impurity_left, double * impurity_right) nogil:
        """
        Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end]).
        """

        cdef DOUBLE_t * sample_weight = self.sample_weight
        cdef SIZE_t * samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double * sum_left = self.sum_left
        cdef double * sum_right = self.sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t is_treated
        cdef DOUBLE_t y_ik

        cdef double right_tr = 0.0
        cdef double right_ct = 0.0
        cdef double right_sum = 0.0
        cdef double right_tr_sum = 0.0
        cdef double right_sq_sum = 0.0
        cdef double right_tr_sq_sum = 0.0
        cdef double right_tr_var
        cdef double right_ct_var

        cdef double left_tr = 0.0
        cdef double left_ct = 0.0
        cdef double left_sum = 0.0
        cdef double left_tr_sum = 0.0
        cdef double left_sq_sum = 0.0
        cdef double left_tr_sq_sum = 0.0
        cdef double left_tr_var
        cdef double left_ct_var

        cdef double eps = 1e-5

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                # It is enough to add eps to get zero values for control
                # treatment: 1 + eps, control: eps
                is_treated = sample_weight[i] - eps

            # assume that there is only one output (k = 0)
            y_ik = self.y[i, 0]

            if p < pos:
                left_tr += is_treated
                left_ct += 1. - is_treated
                left_sum += y_ik
                left_tr_sum += y_ik * is_treated
                left_sq_sum += y_ik * y_ik
                left_tr_sq_sum += y_ik * y_ik * is_treated
            else:
                right_tr += is_treated
                right_ct += 1. - is_treated
                right_sum += y_ik
                right_tr_sum += y_ik * is_treated
                right_sq_sum += y_ik * y_ik
                right_tr_sq_sum += y_ik * y_ik * is_treated

        right_tau = right_tr_sum / right_tr - (sum_right[0] - right_tr_sum) / right_ct
        right_tr_var = right_tr_sq_sum / right_tr - right_tr_sum * right_tr_sum / (right_tr * right_tr)
        right_ct_var = ((right_sq_sum - right_tr_sq_sum) / right_ct -
                        (right_sum - right_tr_sum) * (right_sum - right_tr_sum) / (right_ct * right_ct))

        left_tau = left_tr_sum / left_tr - (sum_left[0] - left_tr_sum) / left_ct
        left_tr_var = left_tr_sq_sum / left_tr - left_tr_sum * left_tr_sum / (left_tr * left_tr)
        left_ct_var = ((left_sq_sum - left_tr_sq_sum) / left_ct -
                       (left_sum - left_tr_sum) * (left_sum - left_tr_sum) / (left_ct * left_ct))

        impurity_left[0] = (left_tr_var / left_tr + left_ct_var / left_ct) - left_tau * left_tau
        impurity_right[0] = (right_tr_var / right_tr + right_ct_var / right_ct) - right_tau * right_tau
