# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


import logging
import numbers
import numpy as np
import pandas as pd

from math import ceil
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.tree._criterion cimport RegressionCriterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._tree import DepthFirstTreeBuilder, DOUBLE, DTYPE, Tree
from sklearn.utils import check_array, check_random_state

from causalml.inference.meta.utils import check_treatment_vector

logger = logging.getLogger('causalml')


cdef class CausalMSE(RegressionCriterion):
    """Causal Tree mean squared error impurity criterion.

        CausalTreeMSE = right_effect + left_effect

        where,

        effect = alpha * tau^2 - (1 - alpha) * (1 + train_to_est_ratio) * (VAR_tr / p + VAR_cont / (1 - p))
    """

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t is_treated
        cdef DOUBLE_t y_ik

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef double node_ct = 0.0
        cdef double node_tr = 0.0
        cdef double node_ct_sum = 0.0
        cdef double node_tr_sum = 0.0
        cdef double one_over_eps = 1e5

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                # the weights of 1 and 1 + eps are used for treatment and control respectively
                is_treated = (sample_weight[i] - 1.0) * one_over_eps

            # assume that there is only one output (k = 0)
            y_ik = self.y[i, 0]

            node_tr += is_treated
            node_ct += 1. - is_treated
            node_tr_sum += y_ik * is_treated
            node_ct_sum += y_ik * (1. - is_treated)

        # save the average of treatment effects within a node as a value for the node
        dest[0] = node_tr_sum / node_tr - node_ct_sum / node_ct

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t is_treated
        cdef DOUBLE_t y_ik

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef double node_tr = 0.0
        cdef double node_ct = 0.0
        cdef double node_sum = self.sum_total[0]
        cdef double node_tr_sum = 0.0
        cdef double node_sq_sum = 0.0
        cdef double node_tr_sq_sum = 0.0
        cdef double tr_var
        cdef double ct_var
        cdef double one_over_eps = 1e5

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                # the weights of 1 and 1 + eps are used for treatment and control respectively
                is_treated = (sample_weight[i] - 1.0) * one_over_eps

            # assume that there is only one output (k = 0)
            y_ik = self.y[i, 0]

            node_tr += is_treated
            node_ct += (1. - is_treated)
            node_tr_sum += y_ik * is_treated
            node_sq_sum += y_ik * y_ik
            node_tr_sq_sum += y_ik * y_ik * is_treated

        node_tau = node_tr_sum / node_tr - (node_sum - node_tr_sum) / node_ct
        tr_var = node_tr_sq_sum / node_tr - node_tr_sum * node_tr_sum / (node_tr * node_tr)
        ct_var = ((node_sq_sum - node_tr_sq_sum) / node_ct -
                  (node_sum - node_tr_sum) * (node_sum - node_tr_sum) / (node_ct * node_ct))

        return  (tr_var / node_tr + ct_var / node_ct) - node_tau * node_tau


    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

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

        cdef double one_over_eps = 1e5

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                # the weights of 1 and 1 + eps are used for control and treatment respectively
                is_treated = (sample_weight[i] - 1.0) * one_over_eps

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


class CausalTreeRegressor:
    """A Causal Tree regressor class.

    The Causal Tree is a decision tree regressor with a split criteria for treatment effects instead of
    outputs.

    Details are available at Athey and Imbens (2015) (https://arxiv.org/abs/1504.01132)
    """
    def __init__(self, ate_alpha=.05, control_name=0, max_depth=None,
                 min_samples_leaf=100, random_state=None):
        """Initialize a Causal Tree

        Args:
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            max_depth (int, optional): the maximum depth of tree
            min_samples_leaf (int, optional): the minimum number of samples in leaves
            random_state (int or np.RandomState, optional): a random seed or a random state
        """
        self.ate_alpha = ate_alpha
        self.control_name = control_name
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self._classes = {}
        self.tree = None

        self.eps = 1e-5

    def fit(self, X, treatment, y):
        """Fit the Causal Tree model

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector

        Returns:
            self (CausalTree object)
        """
        check_treatment_vector(treatment, self.control_name)
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        t_groups = np.unique(treatment[is_treatment])
        self._classes[t_groups[0]] = 0

        # input checking replicated from BaseDecisionTree.fit()
        random_state = check_random_state(self.random_state)
        X = check_array(X, dtype=DTYPE, accept_sparse="csc")
        y = check_array(y, ensure_2d=False, dtype=None)
        if issparse(X):
            X.sort_indices()

            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)

        self.tree = Tree(
            n_features = n_features,
            # line below is taken from DecisionTreeRegressor.fit method source
            #   which comments that the tree shouldn't need the n_classes parameter
            #   but it apparently does
            n_classes = np.array([1] * n_outputs, dtype=np.intp),
            n_outputs = n_outputs)
        splitter = BestSplitter(criterion = CausalMSE(1, X.shape[0]),
            max_features = n_features,
            min_samples_leaf = min_samples_leaf,
            min_weight_leaf = 0, # from DecisionTreeRegressor default
            random_state = random_state)
        # hardcoded values below come from defaults values in
        #   sklearn.tree._classes.DecisionTreeRegressor
        builder = DepthFirstTreeBuilder(
            splitter = splitter,
            min_samples_split = 2,
            min_samples_leaf = min_samples_leaf,
            min_weight_leaf = 0,
            max_depth = max_depth,
            min_impurity_decrease = 0,
            min_impurity_split = float("-inf"))
        builder.build(
            self.tree,
            X = X,
            y = y,
            sample_weight = 1 + self.eps * w)

        return self

    def predict(self, X):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = check_array(X, dtype=DTYPE, accept_sparse="csr")
        return self.tree.predict(X).reshape((-1, 1))

    def fit_predict(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000, verbose=False):
        """Fit the Causal Tree model and predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            verbose (str): whether to output progress logs

        Returns:
           (tuple):

             - te (numpy.ndarray): Predictions of treatment effects.
             - te_lower (numpy.ndarray, optional): lower bounds of treatment effects
             - te_upper (numpy.ndarray, optional): upper bounds of treatment effects
        """
        self.fit(X, treatment, y)
        te = self.predict(X)

        if not return_ci:
            return te
        else:
            start = pd.datetime.today()
            te_bootstraps = np.zeros(shape=(X.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, treatment, y, size=bootstrap_size)
                te_bootstraps[:,i] = np.ravel(te_b)
                if verbose:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds / 60
                    logger.info('{}/{} bootstraps completed. ({:.01f} min lapsed)'.format(i+1, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha/2)*100, axis=1)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha/2)*100, axis=1)

            return (te, te_lower, te_upper)

    def bootstrap(self, X, treatment, y, size=10000):
        """Runs a single bootstrap.

        Fits on bootstrapped sample, then predicts on whole population.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            size (int, optional): bootstrap sample size

        Returns:
            (np.array): bootstrap predictions
        """
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X)
        return te_b

    def estimate_ate(self, X, treatment, y):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector

        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        dhat = self.fit_predict(X, treatment, y)

        te = dhat.mean()
        se = dhat.std() / X.shape[0]

        te_lb = te - se * norm.ppf(1 - self.ate_alpha / 2)
        te_ub = te + se * norm.ppf(1 - self.ate_alpha / 2)

        return te, te_lb, te_ub
