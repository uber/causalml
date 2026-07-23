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

from libc.math cimport log, sqrt, fabs, INFINITY

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

# Weight balancing the parts of the Rzepakowski normalization factor (legacy
# ``arr_normI(..., alpha=0.9)``).
cdef float64_t NORM_ALPHA = 0.9


cdef inline float64_t _entropy_h(float64_t p, float64_t q) noexcept nogil:
    """Legacy ``entropyH(p, q=-1)`` (``uplift.pyx`` ~124).

    Single-argument entropy ``-p*log(p)`` is requested by passing ``q = -1``.
    """
    if q == -1.0 and p > 0.0:
        return -p * log(p)
    elif q > 0.0:
        return -p * log(q)
    else:
        return 0.0


cdef inline float64_t _kl_divergence(float64_t pk, float64_t qk) noexcept nogil:
    """Legacy scalar ``kl_divergence(pk, qk)`` (``uplift.pyx`` ~86).

    Used by the normalization factor for every divergence criterion (KL/ED/Chi),
    independent of the criterion's own ``_pair_divergence``.
    """
    cdef float64_t s

    if qk == 0.0:
        return 0.0

    if qk < EPS:
        qk = EPS
    elif qk > 1.0 - EPS:
        qk = 1.0 - EPS

    if pk == 0.0:
        s = -log(1.0 - qk)
    elif pk == 1.0:
        s = -log(qk)
    else:
        s = pk * log(pk / qk) + (1.0 - pk) * log((1.0 - pk) / (1.0 - qk))
    return s


cdef class UpliftClassificationCriterion(CausalRegressionCriterion):
    """Base class for kernel-backed uplift split criteria."""

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        # The base ``__cinit__`` chain (RegressionCriterion -> CausalRegression
        # Criterion) has already run, so ``self.n_outputs`` and ``self.state``
        # exist. Regularization is off by default; the Python layer sets ``n_reg``
        # / ``min_samples_treatment`` before fitting.
        self.n_reg = 0.0
        self.min_samples_treatment = 0
        self.normalization = 0
        self.has_parent = 0
        self.parent_summary_p.resize(n_outputs, 0.0)
        self._buf_node.resize(n_outputs, 0.0)
        self._buf_left.resize(n_outputs, 0.0)
        self._buf_right.resize(n_outputs, 0.0)

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

    cdef float64_t _node_metric(self, float64_t* p) noexcept nogil:
        """Per-node quantity M in the split gain ``p*M_left + (1-p)*M_right - M_node``.

        Defaults to the summed treatment-vs-control divergence (KL/ED/Chi); CTS
        overrides it with the max over groups.
        """
        return self._divergence_of(p)

    cdef float64_t _split_gain(
        self,
        float64_t* s_node,
        float64_t* s_left,
        float64_t* s_right,
        float64_t p,
    ) noexcept nogil:
        """Split gain from the regularized node / left / right summaries.

        Defaults to ``p*M(left) + (1-p)*M(right) - M(node)`` (the divergence and
        CTS criteria). DDP/IT/CIT override this with their own two-class gain
        forms, reading the raw group counts from ``self.state`` as needed.
        """
        return (
            p * self._node_metric(s_left)
            + (1.0 - p) * self._node_metric(s_right)
            - self._node_metric(s_node)
        )

    cdef bint _normalizable(self) noexcept nogil:
        """Whether Rzepakowski normalization applies to this criterion."""
        return 1

    cdef float64_t _norm_factor(self) noexcept nogil:
        """Rzepakowski normalization factor (legacy ``arr_normI`` else-branch).

        Computed from the *raw* group counts of the current node and one child
        (``uplift.pyx`` ~1673-1733). ``arr_normI`` is asymmetric -- it reads only
        the child legacy calls "left", i.e. the ``col_vals >= value`` partition
        (``group_counts_by_divide`` ~286). The sklearn splitter's left child is
        the complementary ``X <= threshold`` side, so legacy-left is the kernel's
        ``state.right``; the factor must be built from ``state.right``. Always
        >= 0.5, so dividing the gain by it preserves the gain's sign while
        re-ranking splits.
        """
        cdef int32_t i
        cdef float64_t n_c = self.state.node.count_1d[0]
        cdef float64_t n_c_gt = self.state.right.count_1d[0]
        cdef float64_t sum_n_t = 0.0
        cdef float64_t sum_n_t_gt = 0.0
        cdef float64_t n_t_i, pt_a, pc_a, pt_a_i
        cdef float64_t norm_res = 0.0

        for i in range(1, self.n_outputs):
            sum_n_t += self.state.node.count_1d[i]
            sum_n_t_gt += self.state.right.count_1d[i]

        pt_a = sum_n_t_gt / (sum_n_t + 0.1)
        pc_a = n_c_gt / (n_c + 0.1)

        # Part 1
        norm_res += (
            NORM_ALPHA
            * _entropy_h(sum_n_t / (sum_n_t + n_c), n_c / (sum_n_t + n_c))
            * _kl_divergence(pt_a, pc_a)
        )
        # Parts 2 & 3
        for i in range(1, self.n_outputs):
            n_t_i = self.state.node.count_1d[i]
            pt_a_i = self.state.right.count_1d[i] / (n_t_i + 0.1)
            norm_res += (
                (1.0 - NORM_ALPHA)
                * _entropy_h(n_t_i / (n_t_i + n_c), n_c / (n_t_i + n_c))
                * _kl_divergence(pt_a_i, pc_a)
            )
            norm_res += n_t_i / (sum_n_t + n_c) * _entropy_h(pt_a_i, -1.0)
        # Part 4
        norm_res += n_c / (sum_n_t + n_c) * _entropy_h(pc_a, -1.0)
        # Part 5
        norm_res += 0.5
        return norm_res

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

        The gain (:meth:`_split_gain`, criterion-specific) is evaluated on the
        *regularized* summaries: the node shrinks toward its parent, the children
        shrink toward the node. When normalization is enabled (and the criterion
        is normalizable) the gain is divided by :meth:`_norm_factor`. A candidate
        split whose left or right child has any group below
        ``min_samples_treatment`` is inadmissible and gets gain ``-inf`` so the
        split search never selects it.

        ``impurity_left`` / ``impurity_right`` are set to the child metrics for
        interpretability only; the builder reads the gain via
        :meth:`impurity_improvement`.
        """
        cdef int32_t g
        cdef float64_t cnt
        cdef float64_t min_left = self.state.left.count_1d[0]
        cdef float64_t min_right = self.state.right.count_1d[0]
        cdef float64_t* s_node = &self._buf_node[0]
        cdef float64_t* s_left = &self._buf_left[0]
        cdef float64_t* s_right = &self._buf_right[0]
        cdef float64_t p, gain

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
        # Children always shrink toward the (regularized) current node.
        self._reg_summary(self.state.left, s_node, 1, s_left)
        self._reg_summary(self.state.right, s_node, 1, s_right)

        p = self.weighted_n_left / self.weighted_n_node_samples
        gain = self._split_gain(s_node, s_left, s_right, p)
        if self.normalization and self._normalizable():
            gain = gain / self._norm_factor()

        impurity_left[0] = self._node_metric(s_left)
        impurity_right[0] = self._node_metric(s_right)
        self.state.left.split_metric = gain

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


cdef class CTSCriterion(UpliftClassificationCriterion):
    """Contextual Treatment Selection (legacy ``evaluate_CTS`` / ``arr_evaluate_CTS``).

    The node metric is the max over *all* groups (control included) of
    ``P(Y=1|T=g)``, so the gain ``p*max_left + (1-p)*max_right - max_node`` favors
    splits that raise the best achievable outcome. Legacy never normalizes CTS.
    """

    cdef float64_t _node_metric(self, float64_t* p) noexcept nogil:
        cdef float64_t m = p[0]
        cdef int32_t g
        for g in range(1, self.n_outputs):
            if p[g] > m:
                m = p[g]
        return m

    cdef bint _normalizable(self) noexcept nogil:
        return 0


cdef class DDPCriterion(UpliftClassificationCriterion):
    """Delta-Delta-P (legacy ``evaluate_DDP`` / ``arr_evaluate_DDP``).

    Two-class only. The gain is the absolute difference between the children's
    uplift ``DDP(s) = sum_t (P(Y=1|T=t) - P(Y=1|control))``:
    ``|DDP(left) - DDP(right)|``. Legacy never normalizes DDP.
    """

    cdef float64_t _split_gain(
        self,
        float64_t* s_node,
        float64_t* s_left,
        float64_t* s_right,
        float64_t p,
    ) noexcept nogil:
        cdef float64_t d_left = 0.0
        cdef float64_t d_right = 0.0
        cdef int32_t t
        for t in range(1, self.n_outputs):
            d_left += s_left[t] - s_left[0]
            d_right += s_right[t] - s_right[0]
        return fabs(d_left - d_right)

    cdef bint _normalizable(self) noexcept nogil:
        return 0


cdef class ITCriterion(UpliftClassificationCriterion):
    """Interaction-Tree squared t-statistic (legacy ``evaluate_IT`` / ``arr_evaluate_IT``).

    Two-class only. The gain is the squared t-statistic of the difference in
    treatment effect between the two children, using each regularized per-group
    rate's Bernoulli variance ``p*(1-p)`` and the raw group counts. Legacy never
    normalizes IT.
    """

    cdef float64_t _split_gain(
        self,
        float64_t* s_node,
        float64_t* s_left,
        float64_t* s_right,
        float64_t p,
    ) noexcept nogil:
        # Control (group 0) and the single treatment (group 1); two-class only.
        cdef float64_t y_l_0 = s_left[0]
        cdef float64_t y_r_0 = s_right[0]
        cdef float64_t y_l_1 = s_left[1]
        cdef float64_t y_r_1 = s_right[1]
        cdef float64_t n_3 = self.state.left.count_1d[0]
        cdef float64_t n_4 = self.state.right.count_1d[0]
        cdef float64_t n_1 = self.state.left.count_1d[1]
        cdef float64_t n_2 = self.state.right.count_1d[1]
        # Bernoulli sample variances.
        cdef float64_t s_1 = y_l_1 * (1.0 - y_l_1)
        cdef float64_t s_2 = y_r_1 * (1.0 - y_r_1)
        cdef float64_t s_3 = y_l_0 * (1.0 - y_l_0)
        cdef float64_t s_4 = y_r_0 * (1.0 - y_r_0)
        cdef float64_t sum_n = (n_1 - 1.0) + (n_2 - 1.0) + (n_3 - 1.0) + (n_4 - 1.0)
        cdef float64_t w_1 = (n_1 - 1.0) / sum_n
        cdef float64_t w_2 = (n_2 - 1.0) / sum_n
        cdef float64_t w_3 = (n_3 - 1.0) / sum_n
        cdef float64_t w_4 = (n_4 - 1.0) / sum_n
        # Pooled estimator of the constant variance.
        cdef float64_t sigma = sqrt(w_1 * s_1 + w_2 * s_2 + w_3 * s_3 + w_4 * s_4)
        cdef float64_t g_s = (
            ((y_l_1 - y_l_0) - (y_r_1 - y_r_0))
            / (sigma * sqrt(1.0 / n_1 + 1.0 / n_2 + 1.0 / n_3 + 1.0 / n_4))
        )
        return g_s * g_s

    cdef bint _normalizable(self) noexcept nogil:
        return 0


cdef class CITCriterion(UpliftClassificationCriterion):
    """Causal-Inference-Tree likelihood-ratio statistic (legacy ``evaluate_CIT`` / ``arr_evaluate_CIT``).

    Two-class only. The gain is the likelihood-ratio test statistic comparing the
    split model to the parent, using each regularized per-group rate's Bernoulli
    SSE ``n*p*(1-p)`` and the raw group counts. Legacy never normalizes CIT.
    """

    cdef float64_t _split_gain(
        self,
        float64_t* s_node,
        float64_t* s_left,
        float64_t* s_right,
        float64_t p,
    ) noexcept nogil:
        cdef float64_t n_l_t_0 = self.state.left.count_1d[0]
        cdef float64_t n_r_t_0 = self.state.right.count_1d[0]
        cdef float64_t n_l_t_1 = self.state.left.count_1d[1]
        cdef float64_t n_r_t_1 = self.state.right.count_1d[1]
        cdef float64_t n_l_t = n_l_t_1 + n_l_t_0
        cdef float64_t n_r_t = n_r_t_1 + n_r_t_0
        cdef float64_t n_t = n_l_t + n_r_t
        cdef float64_t n_t_1 = n_l_t_1 + n_r_t_1
        cdef float64_t n_t_0 = n_l_t_0 + n_r_t_0
        # Bernoulli SSE = n * p * (1 - p) per group.
        cdef float64_t sse_tau_l = (
            n_l_t_0 * s_left[0] * (1.0 - s_left[0])
            + n_l_t_1 * s_left[1] * (1.0 - s_left[1])
        )
        cdef float64_t sse_tau_r = (
            n_r_t_0 * s_right[0] * (1.0 - s_right[0])
            + n_r_t_1 * s_right[1] * (1.0 - s_right[1])
        )
        cdef float64_t sse_tau = (
            n_t_0 * s_node[0] * (1.0 - s_node[0])
            + n_t_1 * s_node[1] * (1.0 - s_node[1])
        )
        # Maximized log-likelihoods.
        cdef float64_t i_tau_l = (
            -(n_l_t / 2.0) * log(n_l_t * sse_tau_l)
            + n_l_t_1 * log(n_l_t_1) + n_l_t_0 * log(n_l_t_0)
        )
        cdef float64_t i_tau_r = (
            -(n_r_t / 2.0) * log(n_r_t * sse_tau_r)
            + n_r_t_1 * log(n_r_t_1) + n_r_t_0 * log(n_r_t_0)
        )
        cdef float64_t i_tau = (
            -(n_t / 2.0) * log(n_t * sse_tau)
            + n_t_1 * log(n_t_1) + n_t_0 * log(n_t_0)
        )
        return 2.0 * (i_tau_l + i_tau_r - i_tau)

    cdef bint _normalizable(self) noexcept nogil:
        return 0
