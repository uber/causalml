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

Regularization (issue #947)
---------------------------
The gain is computed on *regularized* per-group probabilities, following
Rzepakowski et al. (2012) and the legacy ``UpliftTreeClassifier``:

* ``n_reg`` shrinks each group's ``P(Y=1|T=g)`` toward the parent node's
  regularized probability, with weight ``n_reg`` in sample-size units.
* ``min_samples_treatment`` is the shrinkage threshold (a group at/below it falls
  back entirely to the parent's probability) *and* a hard admissibility floor: a
  candidate split whose left/right child has any group below it is rejected (its
  gain is ``-inf``), so the split search never selects it.

The parent's regularized summary is threaded down the tree by the builder (see
``_uplift/_builder.pyx``): it is set via :meth:`set_parent_summary` before each
``node_split`` and read back via :meth:`compute_reg_summary` after ``node_value``.
The stored leaf *prediction* (``node_value``, inherited from the causal
criterion) remains the raw ``P(Y=1|T=g)`` -- matching the legacy tree, which
regularizes only the split evaluation, not the leaf estimate.

Legacy references in ``uplift.pyx``: ``tree_node_summary`` (~1735, the shrinkage
rule), ``uplift_classification_results`` (~1942, the raw leaf estimate),
``kl_divergence`` / ``evaluate_KL`` / ``evaluate_ED`` / ``evaluate_Chi``
(~1032-1184), and the split gain ``p * D_left + (1 - p) * D_right - D_node`` plus
the ``min_samples_treatment`` reject in ``growDecisionTreeFrom`` (~2270-2316).
"""

from libc.math cimport log, INFINITY

from .._tree._typedefs cimport int32_t, intp_t


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

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        # The base ``__cinit__`` chain (RegressionCriterion -> CausalRegression
        # Criterion) has already run, so ``self.n_outputs`` and ``self.state``
        # exist. Regularization is off by default; the Python layer sets ``n_reg``
        # / ``min_samples_treatment`` before fitting.
        self.n_reg = 0.0
        self.min_samples_treatment = 0
        self.has_parent = 0
        self.parent_summary_p.resize(n_outputs, 0.0)
        self._buf_node.resize(n_outputs, 0.0)
        self._buf_child.resize(n_outputs, 0.0)

    cdef float64_t _pair_divergence(self, float64_t p_t, float64_t p_c) noexcept nogil:
        # Overridden by each concrete criterion.
        return 0.0

    cdef float64_t _divergence_of(self, float64_t* p) noexcept nogil:
        """Sum the treatment-vs-control divergence over all treatment groups."""
        cdef float64_t p_c = p[0]
        cdef float64_t d = 0.0
        cdef int32_t t

        for t in range(1, self.n_outputs):
            d += self._pair_divergence(p[t], p_c)
        return d

    cdef void _reg_summary(
        self,
        NodeState node,
        float64_t* target_p,
        bint has_target,
        float64_t* dest,
    ) noexcept nogil:
        """Regularized P(Y=1|T=g) for every group, written into ``dest``.

        Mirrors the legacy ``tree_node_summary`` shrinkage rule: with no target
        (root) use the raw rate; otherwise shrink toward ``target_p`` with weight
        ``n_reg``, unless the group is at/below ``min_samples_treatment``, in which
        case fall back entirely to the target.
        """
        cdef int32_t g
        cdef float64_t n, n_pos

        for g in range(self.n_outputs):
            n = node.count_1d[g]
            n_pos = node.y_sum_1d[g]
            if not has_target:
                dest[g] = n_pos / n if n > 0 else 0.0
            elif n > self.min_samples_treatment:
                dest[g] = (n_pos + target_p[g] * self.n_reg) / (n + self.n_reg)
            else:
                dest[g] = target_p[g]

    cdef void set_parent_summary(self, float64_t* src, bint has_parent) noexcept nogil:
        """Store the parent node's regularized summary for the next split search."""
        cdef int32_t g
        self.has_parent = has_parent
        if has_parent:
            for g in range(self.n_outputs):
                self.parent_summary_p[g] = src[g]

    cdef void compute_reg_summary(self, float64_t* dest) noexcept nogil:
        """This node's regularized summary (shrunk toward the current parent)."""
        self._reg_summary(self.state.node, &self.parent_summary_p[0], self.has_parent, dest)

    cdef float64_t node_impurity(self) noexcept nogil:
        return NODE_IMPURITY

    cdef void children_impurity(
        self,
        float64_t * impurity_left,
        float64_t * impurity_right,
    ) noexcept nogil:
        """Store the split gain in ``state.left.split_metric``.

        gain = p * D_left + (1 - p) * D_right - D_node, where p is the fraction of
        samples routed to the left child and each ``D`` is the divergence of the
        corresponding *regularized* summary: the node shrinks toward its parent,
        the children shrink toward the node. A candidate split whose left or right
        child has any group below ``min_samples_treatment`` is inadmissible and
        gets gain ``-inf`` so the split search never selects it.

        ``impurity_left`` / ``impurity_right`` are set to the child divergences for
        interpretability only; the builder reads the gain via
        :meth:`impurity_improvement`.
        """
        cdef int32_t g
        cdef float64_t cnt
        cdef float64_t min_left = self.state.left.count_1d[0]
        cdef float64_t min_right = self.state.right.count_1d[0]
        cdef float64_t* s_node = &self._buf_node[0]
        cdef float64_t* s_child = &self._buf_child[0]
        cdef float64_t d_node, d_left, d_right, p

        for g in range(1, self.n_outputs):
            cnt = self.state.left.count_1d[g]
            if cnt < min_left:
                min_left = cnt
            cnt = self.state.right.count_1d[g]
            if cnt < min_right:
                min_right = cnt

        if min_left < self.min_samples_treatment or min_right < self.min_samples_treatment:
            impurity_left[0] = 0.0
            impurity_right[0] = 0.0
            self.state.left.split_metric = -INFINITY
            return

        self._reg_summary(self.state.node, &self.parent_summary_p[0], self.has_parent, s_node)
        d_node = self._divergence_of(s_node)

        # Children always shrink toward the (regularized) current node.
        self._reg_summary(self.state.left, s_node, 1, s_child)
        d_left = self._divergence_of(s_child)

        self._reg_summary(self.state.right, s_node, 1, s_child)
        d_right = self._divergence_of(s_child)

        p = self.weighted_n_left / self.weighted_n_node_samples
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
