# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# distutils: language = c++

"""Uplift classification split criteria on top of the shared ``_tree`` kernel.

These criteria reuse the causal criterion's per-group state machine
(:class:`NodeState` / :class:`NodeSplitState` in ``causal/_criterion.pyx``). For a
binary uplift outcome encoded as the causal-style ``(n_samples, n_groups)``
NaN-masked matrix (control at column 0), ``NodeState.outcome_mean(g)`` is exactly
``P(Y=1 | T=g)`` -- which is all the KL / Euclidean / Chi-squared divergences need.

Growth is driven entirely by ``impurity_improvement`` (the uplift split *gain*),
following the ``TTest`` criterion pattern: ``node_impurity`` is a positive
constant so the kernel builder's ``impurity <= EPSILON`` early-leaf never fires
(an uplift node with zero treatment/control divergence must still be allowed to
split), and the real work happens in ``children_impurity`` /
``impurity_improvement``.

Legacy references in ``uplift.pyx``: ``kl_divergence`` (line ~86),
``evaluate_KL`` / ``evaluate_ED`` / ``evaluate_Chi`` (~1032-1184), and the split
gain ``p * D_left + (1 - p) * D_right - D_node`` in ``growDecisionTreeFrom``
(~2308-2316).
"""

from libc.math cimport log

from .._tree._typedefs cimport int32_t


# Matches ``max(0.1 ** 6, ...)`` in the legacy Chi-squared criterion and the
# ``eps`` clamp in the legacy ``kl_divergence``.
cdef float64_t EPS = 1e-6

# Constant node impurity. Uplift trees are grown by maximizing the split gain in
# ``impurity_improvement``; the node's own "impurity" carries no meaning, so we
# return a fixed positive value purely to keep every node splittable (the kernel
# ``DepthFirstTreeBuilder`` turns a node into a leaf when ``impurity <= EPSILON``,
# which would wrongly stop growth at any node whose treatment and control
# outcomes happen to coincide).
cdef float64_t NODE_IMPURITY = 1.0


cdef class UpliftClassificationCriterion(CausalRegressionCriterion):
    """Base class for kernel-backed uplift split criteria."""

    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil:
        # Overridden by each concrete criterion.
        return 0.0

    cdef float64_t _node_divergence(self, NodeState node) noexcept nogil:
        """Sum the treatment-vs-control divergence over all treatment groups.

        Empty groups fall back to ``P(Y=1) = 0`` (mirrors the legacy
        ``n_pos / n if n > 0 else 0`` guard used at split evaluation with
        ``n_reg = 0`` / ``min_samples_treatment = 0``).
        """
        cdef float64_t p_c, p_t
        cdef float64_t d = 0.0
        cdef int32_t t

        if node.count_1d[0] > 0:
            p_c = node.y_sum_1d[0] / node.count_1d[0]
        else:
            p_c = 0.0

        for t in range(1, self.n_outputs):
            if node.count_1d[t] > 0:
                p_t = node.y_sum_1d[t] / node.count_1d[t]
            else:
                p_t = 0.0
            d += self._pair_divergence(p_t, p_c)
        return d

    cdef float64_t node_impurity(self) noexcept nogil:
        return NODE_IMPURITY

    cdef void children_impurity(
        self,
        float64_t * impurity_left,
        float64_t * impurity_right,
    ) noexcept nogil:
        """Store the split gain in ``state.left.split_metric``.

        gain = p * D_left + (1 - p) * D_right - D_node, where p is the fraction of
        samples routed to the left child. ``impurity_left`` / ``impurity_right``
        are set to the child divergences for interpretability only; the builder
        reads the gain via :meth:`impurity_improvement`.
        """
        cdef float64_t p = self.weighted_n_left / self.weighted_n_node_samples
        cdef float64_t d_node = self._node_divergence(self.state.node)
        cdef float64_t d_left = self._node_divergence(self.state.left)
        cdef float64_t d_right = self._node_divergence(self.state.right)

        impurity_left[0] = d_left
        impurity_right[0] = d_right
        self.state.left.split_metric = p * d_left + (1.0 - p) * d_right - d_node

    cdef float64_t impurity_improvement(
        self,
        float64_t impurity_parent,
        float64_t impurity_left,
        float64_t impurity_right,
    ) noexcept nogil:
        return self.state.left.split_metric

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity(&impurity_left, &impurity_right)
        return self.state.left.split_metric


cdef class KLCriterion(UpliftClassificationCriterion):
    """Kullback-Leibler divergence (legacy ``kl_divergence`` / ``evaluate_KL``)."""

    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil:
        cdef float64_t qk, s

        if p_c == 0.0:
            return 0.0

        qk = p_c
        if qk < EPS:
            qk = EPS
        elif qk > 1.0 - EPS:
            qk = 1.0 - EPS

        if p_t == 0.0:
            s = -log(1.0 - qk)
        elif p_t == 1.0:
            s = -log(qk)
        else:
            s = p_t * log(p_t / qk) + (1.0 - p_t) * log((1.0 - p_t) / (1.0 - qk))
        return s


cdef class EDCriterion(UpliftClassificationCriterion):
    """Euclidean distance (legacy ``evaluate_ED``)."""

    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil:
        cdef float64_t diff = p_t - p_c
        return 2.0 * diff * diff


cdef class ChiCriterion(UpliftClassificationCriterion):
    """Chi-squared statistic (legacy ``evaluate_Chi``)."""

    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil:
        cdef float64_t diff_sq = (p_t - p_c) * (p_t - p_c)
        cdef float64_t denom_pc = p_c if p_c > EPS else EPS
        cdef float64_t denom_1_pc = (1.0 - p_c) if (1.0 - p_c) > EPS else EPS
        return diff_sq / denom_pc + diff_sq / denom_1_pc
