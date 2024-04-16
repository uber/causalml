# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
"""
Forest of trees-based ensemble methods for Uplift modeling on Classification
Problem. Those methods include random forests and extremely randomized trees.

The module structure is the following:
- The ``UpliftRandomForestClassifier`` base class implements different
  variants of uplift models based on random forest, with 'fit' and 'predict'
  method.
- The ``UpliftTreeClassifier`` base class implements the uplift trees (without
  Bootstrapping for random forest), this class is called within
  ``UpliftRandomForestClassifier`` for constructing random forest.

"""

# Authors: Zhenyu Zhao <zhenyuz@uber.com>
#          Totte Harinen <totte@uber.com>

import multiprocessing as mp
from collections import defaultdict

import logging
import cython
import numpy as np
cimport numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from joblib import Parallel, delayed
from packaging import version
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y, check_array, check_random_state

if version.parse(sklearn.__version__) >= version.parse('0.22.0'):
    from sklearn.utils._testing import ignore_warnings
else:
    from sklearn.utils.testing import ignore_warnings

N_TYPE = np.int32
TR_TYPE = np.int8
Y_TYPE = np.int8
P_TYPE = np.float64

ctypedef np.int32_t N_TYPE_t
ctypedef np.int8_t TR_TYPE_t
ctypedef np.int8_t Y_TYPE_t
ctypedef np.float64_t P_TYPE_t

MAX_INT = np.iinfo(np.int32).max

logger = logging.getLogger("causalml")

cdef extern from "math.h":
    double log(double x) nogil
    double fabs(double x) nogil
    double sqrt(double x) nogil

@cython.cfunc
def kl_divergence(pk: cython.float, qk: cython.float) -> cython.float:
    '''
    Calculate KL Divergence for binary classification.

    sum(np.array(pk) * np.log(np.array(pk) / np.array(qk)))

    Args
    ----
    pk : float
        The probability of 1 in one distribution.
    qk : float
        The probability of 1 in the other distribution.

    Returns
    -------
    S : float
        The KL divergence.
    '''

    eps: cython.float = 1e-6
    S: cython.float

    if qk == 0.:
        return 0.

    qk = min(max(qk, eps), 1 - eps)

    if pk == 0.:
        S = -log(1 - qk)
    elif pk == 1.:
        S = -log(qk)
    else:
        S = pk * log(pk / qk) + (1 - pk) * log((1 - pk) / (1 - qk))

    return S


@cython.cfunc
def entropyH(p: cython.float, q: cython.float=-1.) -> cython.float:
    '''
    Entropy

    Entropy calculation for normalization.

    Args
    ----
    p : float
        The probability used in the entropy calculation.

    q : float, optional, (default = -1.)
        The second probability used in the entropy calculation.

    Returns
    -------
    entropy : float
    '''

    if q == -1. and p > 0.:
        return -p * log(p)
    elif q > 0.:
        return -p * log(q)
    else:
        return 0.


class DecisionTree:
    """ Tree Node Class

    Tree node class to contain all the statistics of the tree node.

    Parameters
    ----------
    classes_ : list of str
        A list of the control and treatment group names.

    col : int, optional (default = -1)
        The column index for splitting the tree node to children nodes.

    value : float, optional (default = None)
        The value of the feature column to split the tree node to children nodes.

    trueBranch : object of DecisionTree
        The true branch tree node (feature > value).

    falseBranch : object of DecisionTree
        The false branch tree node (feature > value).

    results : list of float
        The classification probability P(Y=1|T) for each of the control and treatment groups
        in the tree node.

    summary : list of list
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.

    maxDiffTreatment : int
        The treatment index generating the maximum difference between the treatment and control groups.

    maxDiffSign : float
        The sign of the maximum difference (1. or -1.).

    nodeSummary : list of list
        Summary statistics of the tree nodes [P(Y=1|T), N(T)], where y_mean stands for the target metric mean
        and n is the sample size.

    backupResults : list of float
        The positive probabilities in each of the control and treatment groups in the parent node. The parent node
        information is served as a backup for the children node, in case no valid statistics can be calculated from the
        children node, the parent node information will be used in certain cases.

    bestTreatment : int
        The treatment index providing the best uplift (treatment effect).

    upliftScore : list
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maximum treatment effect, and
        p_value stands for the p_value of the treatment effect.

    matchScore : float
        The uplift score by filling a trained tree with validation dataset or testing dataset.

    """

    def __init__(self, classes_, col=-1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None,
                  maxDiffTreatment=None, maxDiffSign=1., nodeSummary=None, backupResults=None, bestTreatment=None,
                  upliftScore=None, matchScore=None):
        self.classes_ = classes_
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # None for nodes, not None for leaves
        self.summary = summary
        # the treatment with max( |p(y|treatment) - p(y|control)| )
        self.maxDiffTreatment = maxDiffTreatment
        # the sign for p(y|maxDiffTreatment) - p(y|control)
        self.maxDiffSign = maxDiffSign
        self.nodeSummary = nodeSummary
        self.backupResults = backupResults
        self.bestTreatment = bestTreatment
        self.upliftScore = upliftScore
        # match actual treatment for validation and testing
        self.matchScore = matchScore


def group_uniqueCounts_to_arr(np.ndarray[TR_TYPE_t, ndim=1] treatment_idx,
                              np.ndarray[Y_TYPE_t, ndim=1] y,
                              np.ndarray[N_TYPE_t, ndim=1] out_arr):
    '''
        Count sample size by experiment group.

        Args
        ----
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
            Should be of type numpy.int8
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
            Should be of type numpy.int8
        out_arr : array-like, shape = [2 * n_class]
            An array to store the output counts, should have type numpy.int32

    Returns
    -------

    No return value, but modified the out_arr to hold the negative and positive
    outcome sample sizes for each of the control and treatment groups.
        out_arr[2*i] is N(Y = 0, T = i) for i = 0, ..., n_class
        out_arr[2*i+1] is N(Y = 1, T = i) for i = 0, ..., n_class
    '''
    cdef int out_arr_len = out_arr.shape[0]
    cdef int n_class = out_arr_len / 2
    cdef int num_samples = treatment_idx.shape[0]
    cdef int yv = 0
    cdef int tv = 0
    cdef int i = 0
    # first clear the output
    for i in range(out_arr_len):
        out_arr[i] = 0
    # then loop through treatment_idx and y, sum the counts
    # first sum as N(T = i) and N(Y = 1, T = i) at index (2*i, 2*i+1), and later adjust
    for i in range(num_samples):
        tv = treatment_idx[i]
        # assume treatment index is in range
        out_arr[2*tv] += 1
        # assume y should be either 0 or 1, so this is summing 
        out_arr[2*tv + 1] += y[i]
    # adjust the entry at index 2*i to be N(Y = 0, T = i) = N(T = i) - N(Y = 1, T = i)
    for i in range(n_class):
        out_arr[2*i] -= out_arr[2*i + 1]
    # done, modified out_arr, so no need to return it

def group_counts_by_divide(
        col_vals, threshold_val, is_split_by_gt,
        np.ndarray[TR_TYPE_t, ndim=1] treatment_idx,
        np.ndarray[Y_TYPE_t, ndim=1] y,
        np.ndarray[N_TYPE_t, ndim=1] out_arr):
    '''
    Count sample size by experiment group for the left branch,
    after splitting col_vals by threshold_val.
    If is_split_by_gt, the left branch is (col_vals >= threshold_val),
    otherwise the left branch is (col_vals == threshold_val).

    This aims to combine the previous divideSet_len and
    group_uniqueCounts_to_arr into one function, so as to reduce the
    number of intermediate objects.

    Args
    ----
    col_vals : array-like, shape = [num_samples]
        An array containing one column of x values.
    threshold_val : compatible value with col_vals
        A value for splitting col_vals.
        If is_split_by_gt, the left branch is (col_vals >= threshold_val),
        otherwise the left branch is (col_vals == threshold_val).
    is_split_by_gt : bool
        Whether to split by (col_vals >= threshold_val).
        If False, will split by (col_vals == threshold_val).
    treatment_idx : array-like, shape = [num_samples]
        An array containing the treatment group index for each unit.
        Should be of type numpy.int8
    y : array-like, shape = [num_samples]
        An array containing the outcome of interest for each unit.
        Should be of type numpy.int8
    out_arr : array-like, shape = [2 * n_class]
        An array to store the output counts, should have type numpy.int32

    Returns
    -------
    len_X_l: the number of samples in the left branch.
    Also modify the out_arr to hold the negative and positive
    outcome sample sizes for each of the control and treatment groups.
        out_arr[2*i] is N(Y = 0, T = i) for i = 0, ..., n_class
        out_arr[2*i+1] is N(Y = 1, T = i) for i = 0, ..., n_class
    '''
    cdef int out_arr_len = out_arr.shape[0]
    cdef int n_class = out_arr_len / 2
    cdef int num_samples = treatment_idx.shape[0]
    cdef int yv = 0
    cdef int tv = 0
    cdef int i = 0
    cdef N_TYPE_t len_X_l = 0
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] filt
    # first clear the output
    for i in range(out_arr_len):
        out_arr[i] = 0

    # split
    if is_split_by_gt:
        filt = col_vals >= threshold_val
    else:
        filt = col_vals == threshold_val

    # then loop through treatment_idx and y, sum the counts where filt
    # is True, and it is the count for the left branch.
    # Also count len_X_l in the process.

    # first sum as N(T = i) and N(Y = 1, T = i) at index (2*i, 2*i+1), and later adjust
    for i in range(num_samples):
        if filt[i]> 0:
            len_X_l += 1
            tv = treatment_idx[i]
            # assume treatment index is in range
            out_arr[2*tv] += 1
            # assume y should be either 0 or 1, so this is summing 
            out_arr[2*tv + 1] += y[i]
    # adjust the entry at index 2*i to be N(Y = 0, T = i) = N(T = i) - N(Y = 1, T = i)
    for i in range(n_class):
        out_arr[2*i] -= out_arr[2*i + 1]
    # done, modified out_arr
    return len_X_l

# Uplift Tree Classifier
class UpliftTreeClassifier:
    """ Uplift Tree Classifier for Classification Task.

    A uplift tree classifier estimates the individual treatment effect by modifying the loss function in the
    classification trees.

    The uplift tree classifier is used in uplift random forest to construct the trees in the forest.

    Parameters
    ----------

    evaluationFunction : string
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS', 'DDP', 'IT', 'CIT', 'IDDP'.

    max_features: int, optional (default=None)
        The number of features to consider when looking for the best split.

    max_depth: int, optional (default=3)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=100)
        The regularization parameter defined in Rzepakowski et al. 2012, the weight (in terms of sample size) of the
        parent node influence on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
    
    early_stopping_eval_diff_scale: float, optional (default=1)
        If train and valid uplift score diff bigger than 
        min(train_uplift_score,valid_uplift_score)/early_stopping_eval_diff_scale, stop.

    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups).

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012, correcting for tests with large number of splits
        and imbalanced treatment and control splits.

    honesty: bool (default=False)
         True if the honest approach based on "Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects."
         shall be used. If 'IDDP' is used as evaluation function, this parameter is automatically set to true.

    estimation_sample_size: float (default=0.5)
         Sample size for estimating the CATE score in the leaves if honesty == True.

    random_state: int, RandomState instance or None (default=None)
        A random seed or `np.random.RandomState` to control randomness in building a tree.

    """
    def __init__(self, control_name, max_features=None, max_depth=3, min_samples_leaf=100,
                 min_samples_treatment=10, n_reg=100, early_stopping_eval_diff_scale=1, evaluationFunction='KL',
                 normalization=True, honesty=False, estimation_sample_size=0.5, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.early_stopping_eval_diff_scale = early_stopping_eval_diff_scale
        self.max_features = max_features

        assert evaluationFunction in ['KL', 'ED', 'Chi', 'CTS', 'DDP', 'IT', 'CIT', 'IDDP'], \
            f"evaluationFunction should be either 'KL', 'ED', 'Chi', 'CTS', 'DDP', 'IT', 'CIT', or 'IDDP' but {evaluationFunction} is passed"

        if evaluationFunction == 'KL':
            self.evaluationFunction = self.evaluate_KL
            self.arr_eval_func = self.arr_evaluate_KL
        elif evaluationFunction == 'ED':
            self.evaluationFunction = self.evaluate_ED
            self.arr_eval_func = self.arr_evaluate_ED
        elif evaluationFunction == 'Chi':
            self.evaluationFunction = self.evaluate_Chi
            self.arr_eval_func = self.arr_evaluate_Chi     
        elif evaluationFunction == 'DDP':
            self.evaluationFunction = self.evaluate_DDP
            self.arr_eval_func = self.arr_evaluate_DDP
        elif evaluationFunction == 'IT':
            self.evaluationFunction = self.evaluate_IT
            self.arr_eval_func = self.arr_evaluate_IT
        elif evaluationFunction == 'CIT':
            self.evaluationFunction = self.evaluate_CIT
            self.arr_eval_func = self.arr_evaluate_CIT
        elif evaluationFunction == 'IDDP':
            self.evaluationFunction = self.evaluate_IDDP
            self.arr_eval_func = self.arr_evaluate_IDDP
        elif evaluationFunction == 'CTS':
            self.evaluationFunction = self.evaluate_CTS
            self.arr_eval_func = self.arr_evaluate_CTS
        self.fitted_uplift_tree = None

        assert control_name is not None and isinstance(control_name, str), \
            f"control_group should be string but {control_name} is passed"
        self.control_name = control_name
        self.classes_ = [self.control_name]
        self.n_class = 1
        self.normalization = normalization
        self.honesty = honesty
        self.estimation_sample_size = estimation_sample_size
        self.random_state = random_state
        if evaluationFunction == 'IDDP' and self.honesty is False:
            self.honesty = True


    def fit(self, X, treatment, y, X_val=None, treatment_val=None, y_val=None):
        """ Fit the uplift model.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.

        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        self : object
        """

        self.random_state_ = check_random_state(self.random_state)

        X, y = check_X_y(X, y)
        y = (y > 0).astype(Y_TYPE) # make sure it is 0 or 1, and is int8
        treatment = np.asarray(treatment)
        assert len(y) == len(treatment), 'Data length must be equal for X, treatment, and y.'
        if X_val is not None:
            X_val, y_val = check_X_y(X_val, y_val)
            y_val = (y_val > 0).astype(Y_TYPE) # make sure it is 0 or 1, and is int8
            treatment_val = np.asarray(treatment_val)
            assert len(y_val) == len(treatment_val), 'Data length must be equal for X_val, treatment_val, and y_val.'
        
        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_groups = sorted([x for x in list(set(treatment)) if x != self.control_name])
        self.classes_ = [self.control_name]
        treatment_idx = np.zeros_like(treatment, dtype=TR_TYPE)
        treatment_val_idx = None
        if treatment_val is not None:
            treatment_val_idx = np.zeros_like(treatment_val, dtype=TR_TYPE)
        for i, tr in enumerate(treatment_groups, 1):
            self.classes_.append(tr)
            treatment_idx[treatment == tr] = i
            if treatment_val_idx is not None:
                treatment_val_idx[treatment_val == tr] = i
        self.n_class = len(self.classes_)

        self.feature_imp_dict = defaultdict(float)

        if (self.n_class > 2) and (self.evaluationFunction in [self.evaluate_DDP, self.evaluate_IDDP, self.evaluate_IT, self.evaluate_CIT]):
            raise ValueError("The DDP, IDDP, IT, and CIT approach can only cope with two class problems, that is two different treatment "
                             "options (e.g., control vs treatment). Please select another approach or only use a "
                             "dataset which employs two treatment options.")

        if self.honesty:
            try:
                X, X_est, treatment_idx, treatment_idx_est, y, y_est = train_test_split(X, treatment_idx, y, stratify=np.stack([treatment_idx, y], axis=1), test_size=self.estimation_sample_size,
                                                                                        shuffle=True, random_state=self.random_state)
            except ValueError:
                logger.warning(f"Stratified sampling failed. Falling back to random sampling.")
                X, X_est, treatment_idx, treatment_idx_est, y, y_est = train_test_split(X, treatment_idx, y, test_size=self.estimation_sample_size, shuffle=True,
                                                                                        random_state=self.random_state)

        self.fitted_uplift_tree = self.growDecisionTreeFrom(
            X, treatment_idx, y, X_val, treatment_val_idx, y_val,
            max_depth=self.max_depth, early_stopping_eval_diff_scale=self.early_stopping_eval_diff_scale,
            min_samples_leaf=self.min_samples_leaf,
            depth=1, min_samples_treatment=self.min_samples_treatment,
            n_reg=self.n_reg, parentNodeSummary_p=None
        )

        if self.honesty:
            self.honestApproach(X_est, treatment_idx_est, y_est)

        self.feature_importances_ = np.zeros(X.shape[1])
        for col, imp in self.feature_imp_dict.items():
            self.feature_importances_[col] = imp
        self.feature_importances_ /= self.feature_importances_.sum()  # normalize to add to 1

    # Prune Trees
    def prune(self, X, treatment, y, minGain=0.0001, rule='maxAbsDiff'):
        """ Prune the uplift model.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        minGain : float, optional (default = 0.0001)
            The minimum gain required to make a tree node split. The children
            tree branches are trimmed if the actual split gain is less than
            the minimum gain.
        rule : string, optional (default = 'maxAbsDiff')
            The prune rules. Supported values are 'maxAbsDiff' for optimizing
            the maximum absolute difference, and 'bestUplift' for optimizing
            the node-size weighted treatment effect.
        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y)
        treatment = np.asarray(treatment)
        assert len(y) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_idx = np.zeros_like(treatment)
        for i, tr in enumerate(self.classes_[1:], 1):
            treatment_idx[treatment == tr] = i

        self.pruneTree(X, treatment_idx, y,
                       tree=self.fitted_uplift_tree,
                       rule=rule,
                       minGain=minGain,
                       n_reg=self.n_reg,
                       parentNodeSummary=None)
        return self

    def honestApproach(self, X_est, T_est, Y_est):
        """ Apply the honest approach based on "Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects."
        Args
        ----
        X_est : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to calculate the unbiased estimates in the leafs of the decision tree.
        T_est : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        Y_est : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        """

        self.fillTree(X_est, T_est, Y_est, self.fitted_uplift_tree)

    def pruneTree(self, X, treatment_idx, y, tree, rule='maxAbsDiff', minGain=0.,
                  n_reg=0,
                  parentNodeSummary=None):
        """Prune one single tree node in the uplift model.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        rule : string, optional (default = 'maxAbsDiff')
            The prune rules. Supported values are 'maxAbsDiff' for optimizing the maximum absolute difference, and
            'bestUplift' for optimizing the node-size weighted treatment effect.
        minGain : float, optional (default = 0.)
            The minimum gain required to make a tree node split. The children tree branches are trimmed if the actual
            split gain is less than the minimum gain.
        n_reg: int, optional (default=0)
            The regularization parameter defined in Rzepakowski et al. 2012, the weight (in terms of sample size) of the
            parent node influence on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : list of list, optional (default = None)
            Node summary statistics, [P(Y=1|T), N(T)] of the parent tree node.
        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(
            treatment_idx, y, min_samples_treatment=self.min_samples_treatment,
            n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        if (tree.trueBranch is None) or (tree.falseBranch is None):
            X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, tree.col, tree.value)

            # recursive call for each branch
            if tree.trueBranch.results is None:
                self.pruneTree(X_l, w_l, y_l, tree.trueBranch, rule, minGain,
                               n_reg,
                               parentNodeSummary=currentNodeSummary)
            if tree.falseBranch.results is None:
                self.pruneTree(X_r, w_r, y_r, tree.falseBranch, rule, minGain,
                               n_reg,
                               parentNodeSummary=currentNodeSummary)

        # merge leaves (potentially)
        if (tree.trueBranch.results is not None and
            tree.falseBranch.results is not None):
            if rule == 'maxAbsDiff':
                # Current D
                if (tree.maxDiffTreatment in currentNodeSummary and
                    self.control_name in currentNodeSummary):
                    currentScoreD = tree.maxDiffSign * (currentNodeSummary[tree.maxDiffTreatment][0]
                                                        - currentNodeSummary[self.control_name][0])
                else:
                    currentScoreD = 0

                # trueBranch D
                trueNodeSummary = self.tree_node_summary(
                    w_l, y_l, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.trueBranch.maxDiffTreatment in trueNodeSummary and
                    self.control_name in trueNodeSummary):
                    trueScoreD = tree.trueBranch.maxDiffSign * (trueNodeSummary[tree.trueBranch.maxDiffTreatment][0]
                                                                - trueNodeSummary[self.control_name][0])
                    trueScoreD = (
                        trueScoreD
                        * (trueNodeSummary[tree.trueBranch.maxDiffTreatment][1]
                         + trueNodeSummary[self.control_name][1])
                        / (currentNodeSummary[tree.trueBranch.maxDiffTreatment][1]
                           + currentNodeSummary[self.control_name][1])
                    )
                else:
                    trueScoreD = 0

                # falseBranch D
                falseNodeSummary = self.tree_node_summary(
                    w_r, y_r, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.falseBranch.maxDiffTreatment in falseNodeSummary and
                    self.control_name in falseNodeSummary):
                    falseScoreD = (
                        tree.falseBranch.maxDiffSign *
                        (falseNodeSummary[tree.falseBranch.maxDiffTreatment][0]
                         - falseNodeSummary[self.control_name][0])
                    )

                    falseScoreD = (
                        falseScoreD *
                        (falseNodeSummary[tree.falseBranch.maxDiffTreatment][1]
                         + falseNodeSummary[self.control_name][1])
                        / (currentNodeSummary[tree.falseBranch.maxDiffTreatment][1]
                           + currentNodeSummary[self.control_name][1])
                    )
                else:
                    falseScoreD = 0

                if ((trueScoreD + falseScoreD) - currentScoreD <= minGain or
                    (trueScoreD + falseScoreD < 0.)):
                    tree.trueBranch, tree.falseBranch = None, None
                    tree.results = tree.backupResults

            elif rule == 'bestUplift':
                # Current D
                if (tree.bestTreatment in currentNodeSummary and
                    self.control_name in currentNodeSummary):
                    currentScoreD = (
                        currentNodeSummary[tree.bestTreatment][0]
                        - currentNodeSummary[self.control_name][0]
                    )
                else:
                    currentScoreD = 0

                # trueBranch D
                trueNodeSummary = self.tree_node_summary(
                    w_l, y_l, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.trueBranch.bestTreatment in trueNodeSummary and
                    self.control_name in trueNodeSummary):
                    trueScoreD = (
                        trueNodeSummary[tree.trueBranch.bestTreatment][0]
                        - trueNodeSummary[self.control_name][0]
                    )
                else:
                    trueScoreD = 0

                # falseBranch D
                falseNodeSummary = self.tree_node_summary(
                    w_r, y_r, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.falseBranch.bestTreatment in falseNodeSummary and
                    self.control_name in falseNodeSummary):
                    falseScoreD = (
                        falseNodeSummary[tree.falseBranch.bestTreatment][0]
                        - falseNodeSummary[self.control_name][0]
                    )
                else:
                    falseScoreD = 0
                gain = ((1. * len(y_l) / len(y) * trueScoreD
                         + 1. * len(y_r) / len(y) * falseScoreD)
                        - currentScoreD)
                if gain <= minGain or (trueScoreD + falseScoreD < 0.):
                    tree.trueBranch, tree.falseBranch = None, None
                    tree.results = tree.backupResults
        return self

    def fill(self, X, treatment, y):
        """ Fill the data into an existing tree.
        This is a higher-level function to transform the original data inputs
        into lower level data inputs (list of list and tree).

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y)
        treatment = np.asarray(treatment)
        assert len(y) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_idx = np.zeros_like(treatment)
        for i, tr in enumerate(self.classes_[1:], 1):
            treatment_idx[treatment == tr] = i

        self.fillTree(X, treatment_idx, y, tree=self.fitted_uplift_tree)
        return self

    def fillTree(self, X, treatment_idx, y, tree):
        """ Fill the data into an existing tree.
        This is a lower-level function to execute on the tree filling task.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        tree : object
            object of DecisionTree class

        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(treatment_idx, y,
                                                    min_samples_treatment=0,
                                                    n_reg=0,
                                                    parentNodeSummary=None)
        tree.nodeSummary = currentNodeSummary

        # Divide sets for child nodes
        if tree.trueBranch or tree.falseBranch:
            X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, tree.col, tree.value)

            # recursive call for each branch
            if tree.trueBranch is not None:
                self.fillTree(X_l, w_l, y_l, tree.trueBranch)
            if tree.falseBranch is not None:
                self.fillTree(X_r, w_r, y_r, tree.falseBranch)

        # Update Information

        # matchScore
        matchScore = (currentNodeSummary[tree.bestTreatment][0] - currentNodeSummary[0][0])
        tree.matchScore = round(matchScore, 4)
        tree.summary['matchScore'] = round(matchScore, 4)

        # Samples, Group_size
        tree.summary['samples'] = len(y)
        tree.summary['group_size'] = ''
        for treatment_group, summary in zip(self.classes_, currentNodeSummary):
            tree.summary['group_size'] += ' ' + treatment_group + ': ' + str(summary[1])
        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(treatment_idx, y)
        return self

    def predict(self, X):
        '''
        Returns the recommended treatment group and predicted optimal
        probability conditional on using the recommended treatment group.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        Returns
        -------
        pred: ndarray, shape = [num_samples, num_treatments]
            An ndarray of predicted treatment effects across treatments.
        '''

        X = check_array(X)

        pred_nodes = []
        for i_row in range(len(X)):
            pred_leaf, _ = self.classify(X[i_row], self.fitted_uplift_tree, dataMissing=False)
            pred_nodes.append(pred_leaf)
        return np.array(pred_nodes)

    @staticmethod
    def divideSet(X, treatment_idx, y, column, value):
        '''
        Tree node split.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        column : int
                The column used to split the data.
        value : float or int
                The value in the column for splitting the data.

        Returns
        -------
        (X_l, X_r, treatment_l, treatment_r, y_l, y_r) : list of ndarray
                The covariates, treatments and outcomes of left node and the right node.
        '''
        # for int and float values
        if np.issubdtype(value.dtype, np.number):
            filt = X[:, column] >= value
        else:  # for strings
            filt = X[:, column] == value

        return X[filt], X[~filt], treatment_idx[filt], treatment_idx[~filt], y[filt], y[~filt]

    @staticmethod
    def divideSet_len(X, treatment_idx, y, column, value):
        '''Tree node split.

        Modified from dividedSet(), but return the len(X_l) and
        len(X_r) instead of the split X_l and X_r, to avoid some
        overhead, intended to be used for finding the split. After
        finding the best splits, can split to find the X_l and X_r.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        column : int
                The column used to split the data.
        value : float or int
                The value in the column for splitting the data.

        Returns
        -------
        (len_X_l, len_X_r, treatment_l, treatment_r, y_l, y_r) : list of ndarray
                The covariates nrows, treatments and outcomes of left node and the right node.

        '''
        # for int and float values
        if np.issubdtype(value.dtype, np.number):
            filt = X[:, column] >= value
        else:  # for strings
            filt = X[:, column] == value

        len_X_l = np.sum(filt)
        return len_X_l, len(X) - len_X_l, treatment_idx[filt], treatment_idx[~filt], y[filt], y[~filt]

    def group_uniqueCounts(self, treatment_idx, y):
        '''
        Count sample size by experiment group.

        Args
        ----
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        results : list of list
            The negative and positive outcome sample sizes for each of the control and treatment groups.
        '''
        results = []
        for i in range(self.n_class):
            filt = treatment_idx == i
            n_pos = y[filt].sum()

            # [N(Y = 0, T = 1), N(Y = 1, T = 1)]
            results.append([filt.sum() - n_pos, n_pos])

        return results

    @staticmethod
    def evaluate_KL(nodeSummary):
        '''
        Calculate KL Divergence as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary()
            method.

        Returns
        -------
        d_res : KL Divergence
        '''
        p_c = nodeSummary[0][0]
        d_res = 0.
        for treatment_group in nodeSummary[1:]:
            d_res += kl_divergence(treatment_group[0], p_c)
        return d_res

    @staticmethod
    def arr_evaluate_KL(np.ndarray[P_TYPE_t, ndim=1] node_summary_p,
                        np.ndarray[N_TYPE_t, ndim=1] node_summary_n):
        '''
        Calculate KL Divergence as split evaluation criterion for a given node.
        Modified to accept new node summary format.

        Args
        ----
        node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        Returns
        -------
        d_res : KL Divergence
        '''
        cdef int n_class = node_summary_p.shape[0]
        cdef P_TYPE_t p_c = node_summary_p[0]
        cdef P_TYPE_t d_res = 0.0
        cdef int i = 0
        for i in range(1, n_class):
            d_res += kl_divergence(node_summary_p[i], p_c)
        return d_res

    @staticmethod
    def evaluate_ED(nodeSummary):
        '''
        Calculate Euclidean Distance as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary()
            method.

        Returns
        -------
        d_res : Euclidean Distance
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:]:
            d_res += 2*(treatment_group[0] - pc)**2
        return d_res

    @staticmethod
    def arr_evaluate_ED(np.ndarray[P_TYPE_t, ndim=1] node_summary_p,
                        np.ndarray[N_TYPE_t, ndim=1] node_summary_n):
        '''
        Calculate Euclidean Distance as split evaluation criterion for a given node.

        Args
        ----
        node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        Returns
        -------
        d_res : Euclidean Distance
        '''
        cdef int n_class = node_summary_p.shape[0]
        cdef P_TYPE_t p_c = node_summary_p[0]
        cdef P_TYPE_t d_res = 0.0
        cdef int i = 0
        for i in range(1, n_class):
            d_res += 2*(node_summary_p[i] - p_c)*(node_summary_p[i] - p_c)
        return d_res

    @staticmethod
    def evaluate_Chi(nodeSummary):
        '''
        Calculate Chi-Square statistic as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.

        Returns
        -------
        d_res : Chi-Square
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:]:
            d_res += ((treatment_group[0] - pc) ** 2 / max(0.1 ** 6, pc)
                      + (treatment_group[0] - pc) ** 2 / max(0.1 ** 6, 1 - pc))
        return d_res

    @staticmethod
    def arr_evaluate_Chi(np.ndarray[P_TYPE_t, ndim=1] node_summary_p,
                         np.ndarray[N_TYPE_t, ndim=1] node_summary_n):
        '''
        Calculate Chi-Square statistic as split evaluation criterion for a given node.

        Args
        ----
        node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        Returns
        -------
        d_res : Chi-Square
        '''
        cdef int n_class = node_summary_p.shape[0]
        cdef P_TYPE_t p_c = node_summary_p[0]
        cdef P_TYPE_t d_res = 0.0
        cdef int i = 0
        cdef P_TYPE_t max_eps_pc = max(0.1 ** 6, p_c)
        cdef P_TYPE_t max_eps_1_pc = max(0.1 ** 6, 1 - p_c)
        cdef P_TYPE_t diff_sq = 0.0
        for i in range(1, n_class):
            diff_sq = (node_summary_p[i] - p_c) * (node_summary_p[i] - p_c)
            d_res += (diff_sq / max_eps_pc + diff_sq / max_eps_1_pc)
        return d_res

    @staticmethod
    def evaluate_DDP(nodeSummary):
        '''
        Calculate Delta P as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary() method.

        Returns
        -------
        d_res : Delta P
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:]:
            d_res += treatment_group[0] - pc
        return d_res

    @staticmethod
    def arr_evaluate_DDP(np.ndarray[P_TYPE_t, ndim=1] node_summary_p,
                         np.ndarray[N_TYPE_t, ndim=1] node_summary_n):
        '''
        Calculate Delta P as split evaluation criterion for a given node.

        Args
        ----
        node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        Returns
        -------
        d_res : Delta P
        '''
        cdef int n_class = node_summary_p.shape[0]
        cdef P_TYPE_t p_c = node_summary_p[0]
        cdef P_TYPE_t d_res = 0.0
        cdef int i = 0
        for i in range(1, n_class):
            d_res += node_summary_p[i] - p_c
        return d_res

    @staticmethod
    def evaluate_IT(leftNodeSummary, rightNodeSummary, w_l, w_r):
        '''
        Calculate Squared T-Statistic as split evaluation criterion for a given node

        Args
        ----
        leftNodeSummary : list of list
            The left node summary statistics.
        rightNodeSummary : list of list
            The right node summary statistics.
        w_l: array-like, shape = [num_samples]
            An array containing the treatment for each unit in the left node
        w_r: array-like, shape = [num_samples]
            An array containing the treatment for each unit in the right node

        Returns
        -------
        g_s : Squared T-Statistic
        '''
        g_s = 0

        ## Control Group
        # Sample mean in left & right child node
        y_l_0 = leftNodeSummary[0][0]
        y_r_0 = rightNodeSummary[0][0]
        # Sample size left & right child node
        n_3 = leftNodeSummary[0][1]
        n_4 = rightNodeSummary[0][1]
        # Sample variance in left & right child node (p*(p-1) for bernoulli)
        s_3 = y_l_0*(1-y_l_0)
        s_4 = y_r_0*(1-y_r_0)

        for treatment_left, treatment_right in zip(leftNodeSummary[1:], rightNodeSummary[1:]):
            ## Treatment Group
            # Sample mean in left & right child node
            y_l_1 = treatment_left[0]
            y_r_1 = treatment_right[0]
            # Sample size left & right child node
            n_1 = treatment_left[1]
            n_2 = treatment_right[1]
            # Sample variance in left & right child node
            s_1 = y_l_1*(1-y_l_1)
            s_2 = y_r_1*(1-y_r_1)

            sum_n = np.sum([n_1 - 1, n_2 - 1, n_3 - 1, n_4 - 1])
            w_1 = (n_1 - 1) / sum_n
            w_2 = (n_2 - 1) / sum_n
            w_3 = (n_3 - 1) / sum_n
            w_4 = (n_4 - 1) / sum_n

            # Pooled estimator of the constant variance
            sigma = np.sqrt(np.sum([w_1 * s_1, w_2 * s_2, w_3 * s_3, w_4 * s_4]))

            # Squared t-statistic
            g_s = np.power(((y_l_1 - y_l_0) - (y_r_1 - y_r_0)) / (sigma * np.sqrt(np.sum([1 / n_1, 1 / n_2, 1 / n_3, 1 / n_4]))), 2)

        return g_s

    @staticmethod
    def arr_evaluate_IT(np.ndarray[P_TYPE_t, ndim=1] left_node_summary_p,
                        np.ndarray[N_TYPE_t, ndim=1] left_node_summary_n,
                        np.ndarray[P_TYPE_t, ndim=1] right_node_summary_p,
                        np.ndarray[N_TYPE_t, ndim=1] right_node_summary_n):
        '''
        Calculate Squared T-Statistic as split evaluation criterion for a given node

        NOTE: n_class should be 2.

        Args
        ----
        left_node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the left node, i.e. [P(Y=1|T=i)...]
        left_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the left node, i.e. [N(T=i)...]
        right_node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the right node, i.e. [P(Y=1|T=i)...]
        right_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the right node, i.e. [N(T=i)...]

        Returns
        -------
        g_s : Squared T-Statistic
        '''
        ## Control Group
        # Sample mean in left & right child node
        cdef P_TYPE_t y_l_0 = left_node_summary_p[0]
        cdef P_TYPE_t y_r_0 = right_node_summary_p[0]
        # Sample size left & right child node
        cdef N_TYPE_t n_3 = left_node_summary_n[0]
        cdef N_TYPE_t n_4 = right_node_summary_n[0]
        # Sample variance in left & right child node (p*(p-1) for bernoulli)
        cdef P_TYPE_t s_3 = y_l_0*(1-y_l_0)
        cdef P_TYPE_t s_4 = y_r_0*(1-y_r_0)

        # only one treatment, contrast with control, so no need to loop
        ## Treatment Group
        # Sample mean in left & right child node
        cdef P_TYPE_t y_l_1 = left_node_summary_p[1]
        cdef P_TYPE_t y_r_1 = right_node_summary_p[1]
        # Sample size left & right child node
        cdef N_TYPE_t n_1 = left_node_summary_n[1]
        cdef N_TYPE_t n_2 = right_node_summary_n[1]
        # Sample variance in left & right child node
        cdef P_TYPE_t s_1 = y_l_1*(1-y_l_1)
        cdef P_TYPE_t s_2 = y_r_1*(1-y_r_1)

        cdef P_TYPE_t sum_n = (n_1 - 1) + (n_2 - 1) + (n_3 - 1) + (n_4 - 1)
        cdef P_TYPE_t w_1 = (n_1 - 1) / sum_n
        cdef P_TYPE_t w_2 = (n_2 - 1) / sum_n
        cdef P_TYPE_t w_3 = (n_3 - 1) / sum_n
        cdef P_TYPE_t w_4 = (n_4 - 1) / sum_n

        # Pooled estimator of the constant variance
        cdef P_TYPE_t sigma = sqrt(w_1 * s_1 + w_2 * s_2 + w_3 * s_3 + w_4 * s_4)

        # Squared t-statistic
        cdef P_TYPE_t g_s = ((y_l_1 - y_l_0) - (y_r_1 - y_r_0)) / (sigma * sqrt(1.0 / n_1 + 1.0 / n_2 + 1.0 / n_3 + 1.0 / n_4))
        g_s = g_s * g_s

        return g_s

    @staticmethod
    def evaluate_CIT(currentNodeSummary, leftNodeSummary, rightNodeSummary, y_l, y_r, w_l, w_r, y, w):
        '''
        Calculate likelihood ratio test statistic as split evaluation criterion for a given node
        Args
        ----
        currentNodeSummary: list of lists
            The parent node summary statistics
        leftNodeSummary : list of lists
            The left node summary statistics.
        rightNodeSummary : list of lists
            The right node summary statistics.
        y_l: array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit in the left node
        y_r: array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit in the right node
        w_l: array-like, shape = [num_samples]
            An array containing the treatment for each unit in the left node
        w_r: array-like, shape = [num_samples]
            An array containing the treatment for each unit in the right node
        y: array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit
        w: array-like, shape = [num_samples]
            An array containing the treatment for each unit
        Returns
        -------
        lrt : Likelihood ratio test statistic
        '''
        lrt = 0

        # Control sample size left & right child node
        n_l_t_0 = leftNodeSummary[0][1]
        n_r_t_0 = rightNodeSummary[0][1]

        for treatment_left, treatment_right in zip(leftNodeSummary[1:], rightNodeSummary[1:]):
            # Treatment sample size left & right child node
            n_l_t_1 = treatment_left[1]
            n_r_t_1 = treatment_right[1]

            # Total size of left & right node
            n_l_t = n_l_t_1 + n_l_t_0
            n_r_t = n_r_t_1 + n_r_t_0

            # Total size of parent node
            n_t = n_l_t + n_r_t

            # Total treatment & control size in parent node
            n_t_1 = n_l_t_1 + n_r_t_1
            n_t_0 = n_l_t_0 + n_r_t_0

            # Standard squared error of left child node
            sse_tau_l = np.sum(np.power(y_l[w_l == 1] - treatment_left[0], 2)) + np.sum(
                np.power(y_l[w_l == 0] - treatment_left[0], 2))

            # Standard squared error of right child node
            sse_tau_r = np.sum(np.power(y_r[w_r == 1] - treatment_right[0], 2)) + np.sum(
                np.power(y_r[w_r == 0] - treatment_right[0], 2))

            # Standard squared error of parent child node
            sse_tau = np.sum(np.power(y[w == 1] - currentNodeSummary[1][0], 2)) + np.sum(
                np.power(y[w == 0] - currentNodeSummary[0][0], 2))

            # Maximized log-likelihood function
            i_tau_l = - (n_l_t / 2) * np.log(n_l_t * sse_tau_l) + n_l_t_1 * np.log(n_l_t_1) + n_l_t_0 * np.log(n_l_t_0)
            i_tau_r = - (n_r_t / 2) * np.log(n_r_t * sse_tau_r) + n_r_t_1 * np.log(n_r_t_1) + n_r_t_0 * np.log(n_r_t_0)
            i_tau = - (n_t / 2) * np.log(n_t * sse_tau) + n_t_1 * np.log(n_t_1) + n_t_0 * np.log(n_t_0)

            # Likelihood ration test statistic
            lrt = 2 * (i_tau_l + i_tau_r - i_tau)

        return lrt

    @staticmethod
    def arr_evaluate_CIT(np.ndarray[P_TYPE_t, ndim=1] cur_node_summary_p,
                         np.ndarray[N_TYPE_t, ndim=1] cur_node_summary_n,
                         np.ndarray[P_TYPE_t, ndim=1] left_node_summary_p,
                         np.ndarray[N_TYPE_t, ndim=1] left_node_summary_n,
                         np.ndarray[P_TYPE_t, ndim=1] right_node_summary_p,
                         np.ndarray[N_TYPE_t, ndim=1] right_node_summary_n):
        '''
        Calculate likelihood ratio test statistic as split evaluation criterion for a given node
        
        NOTE: n_class should be 2.

        Args
        ----
        cur_node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        cur_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]
        left_node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the left node, i.e. [P(Y=1|T=i)...]
        left_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the left node, i.e. [N(T=i)...]
        right_node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the right node, i.e. [P(Y=1|T=i)...]
        right_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the right node, i.e. [N(T=i)...]
                
        Returns
        -------
        lrt : Likelihood ratio test statistic
        '''
        cdef P_TYPE_t lrt = 0.0

        # since will take log of these N, so use a double type

        # Control sample size left & right child node
        cdef P_TYPE_t n_l_t_0 = left_node_summary_n[0]
        cdef P_TYPE_t n_r_t_0 = right_node_summary_n[0]

        # Treatment sample size left & right child node
        cdef P_TYPE_t n_l_t_1 = left_node_summary_n[1]
        cdef P_TYPE_t n_r_t_1 = right_node_summary_n[1]

        # Total size of left & right node
        cdef P_TYPE_t n_l_t = n_l_t_1 + n_l_t_0
        cdef P_TYPE_t n_r_t = n_r_t_1 + n_r_t_0

        # Total size of parent node
        cdef P_TYPE_t n_t = n_l_t + n_r_t

        # Total treatment & control size in parent node
        cdef P_TYPE_t n_t_1 = n_l_t_1 + n_r_t_1
        cdef P_TYPE_t n_t_0 = n_l_t_0 + n_r_t_0

        # NOTE: the original code for sse_tau_l and sse_tau_r does not seem to follow the paper.
        # sse = \sum_{i for treatment} (y_i - p_treatment)^2 + \sum_{i for control} (y_i - p_control)^2

        # NOTE: since for classification, the y is either 0 or 1, we can calculate sse more simply
        # for y being 0 or 1, sse = n*p*(1-p), but here need to calculate separately for treatment and control groups.

        # Standard squared error of left child node
        cdef P_TYPE_t sse_tau_l = n_l_t_0 * left_node_summary_p[0] * (1.0 - left_node_summary_p[0]) + n_l_t_1 * left_node_summary_p[1] * (1.0 - left_node_summary_p[1])

        # Standard squared error of right child node
        cdef P_TYPE_t sse_tau_r = n_r_t_0 * right_node_summary_p[0] * (1.0 - right_node_summary_p[0]) + n_r_t_1 * right_node_summary_p[1] * (1.0 - right_node_summary_p[1])

        # Standard squared error of parent child node
        cdef P_TYPE_t sse_tau = n_t_0 * cur_node_summary_p[0] * (1.0 - cur_node_summary_p[0]) + n_t_1 * cur_node_summary_p[1] * (1.0 - cur_node_summary_p[1])

        # Maximized log-likelihood function
        cdef P_TYPE_t i_tau_l = - (n_l_t / 2.0) * log(n_l_t * sse_tau_l) + n_l_t_1 * log(n_l_t_1) + n_l_t_0 * log(n_l_t_0)
        cdef P_TYPE_t i_tau_r = - (n_r_t / 2.0) * log(n_r_t * sse_tau_r) + n_r_t_1 * log(n_r_t_1) + n_r_t_0 * log(n_r_t_0)
        cdef P_TYPE_t i_tau = - (n_t / 2.0) * log(n_t * sse_tau) + n_t_1 * log(n_t_1) + n_t_0 * log(n_t_0)

        # Likelihood ration test statistic
        lrt = 2 * (i_tau_l + i_tau_r - i_tau)

        return lrt

    @staticmethod
    def evaluate_IDDP(nodeSummary):
        '''
        Calculate Delta P as split evaluation criterion for a given node.
        
        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.
        control_name : string
            The control group name.
        Returns
        -------
        d_res : Delta P
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:]:
            d_res += treatment_group[0] - pc
        return d_res

    @staticmethod
    def arr_evaluate_IDDP(np.ndarray[P_TYPE_t, ndim=1] node_summary_p,
                          np.ndarray[N_TYPE_t, ndim=1] node_summary_n):
        '''
        Calculate Delta P as split evaluation criterion for a given node.
        
        Args
        ----
        node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        Returns
        -------
        d_res : Delta P
        '''
        cdef int n_class = node_summary_p.shape[0]
        cdef P_TYPE_t p_c = node_summary_p[0]
        cdef P_TYPE_t d_res = 0.0
        cdef int i = 0
        for i in range(1, n_class):
            d_res += node_summary_p[i] - p_c
        return d_res

    @staticmethod
    def evaluate_CTS(nodeSummary):
        '''
        Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary() method.

        Returns
        -------
        d_res : CTS score
        '''
        return -max([stat[0] for stat in nodeSummary])

    @staticmethod
    def arr_evaluate_CTS(np.ndarray[P_TYPE_t, ndim=1] node_summary_p,
                         np.ndarray[N_TYPE_t, ndim=1] node_summary_n):
        '''
        Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.

        Args
        ----
        node_summary_p : array of shape [n_class]
            Has type numpy.double.
            The positive probabilities of each of the control
            and treament groups of the current node, i.e. [P(Y=1|T=i)...]
        node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        Returns
        -------
        d_res : CTS score
        '''
        # not sure why use negative for CTS, but in calculating the
        # gain, it is adjusted back so as to maximize the gain.
        cdef int n_class = node_summary_p.shape[0]
        cdef P_TYPE_t d_res = node_summary_p[0]
        cdef int i = 0
        for i in range(1, n_class):
            if node_summary_p[i] > d_res:
                d_res = node_summary_p[i]
        return -d_res

    def normI(self, n_c: cython.int, n_c_left: cython.int, n_t: list, n_t_left: list, alpha: cython.float = 0.9, currentDivergence: cython.float = 0.0) -> cython.float:
        '''
        Normalization factor.

        Args
        ----
        currentNodeSummary : list of list
            The summary statistics of the current tree node, [P(Y=1|T), N(T)].

        leftNodeSummary : list of list
            The summary statistics of the left tree node, [P(Y=1|T), N(T)].

        alpha : float
            The weight used to balance different normalization parts.

        Returns
        -------
        norm_res : float
            Normalization factor.
        '''

        norm_res: cython.float = 0.
        pt_a: cython.float
        pc_a: cython.float

        pt_a = 1. * np.sum(n_t_left) / (np.sum(n_t) + 0.1)
        pc_a = 1. * n_c_left / (n_c + 0.1)

        if self.evaluationFunction == self.evaluate_IDDP:
            # Normalization Part 1
            norm_res += (entropyH(1. * np.sum(n_t) / (np.sum(n_t) + n_c), 1. * n_c / (np.sum(n_t) + n_c)) * currentDivergence)
            norm_res += (1. * np.sum(n_t) / (np.sum(n_t) + n_c) * entropyH(pt_a))

        else:
            # Normalization Part 1
            norm_res += (alpha * entropyH(1. * np.sum(n_t) / (np.sum(n_t) + n_c), 1. * n_c / (np.sum(n_t) + n_c)) * kl_divergence(pt_a, pc_a))
            # Normalization Part 2 & 3
            for i in range(len(n_t)):
                pt_a_i = 1. * n_t_left[i] / (n_t[i] + 0.1)
                norm_res += ((1 - alpha) * entropyH(1. * n_t[i] / (n_t[i] + n_c), 1. * n_c / (n_t[i] + n_c)) * kl_divergence(1. * pt_a_i, pc_a))
                norm_res += (1. * n_t[i] / (np.sum(n_t) + n_c) * entropyH(pt_a_i))
        # Normalization Part 4
        norm_res += 1. * n_c / (np.sum(n_t) + n_c) * entropyH(pc_a)

        # Normalization Part 5
        norm_res += 0.5
        return norm_res

    def arr_normI(self, cur_node_summary_n, left_node_summary_n,
                  alpha: cython.float = 0.9, currentDivergence: cython.float = 0.0) -> cython.float:
        '''
        Normalization factor.

        Args
        ----
        cur_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the current node, i.e. [N(T=i)...]

        left_node_summary_n : array of shape [n_class]
            Has type numpy.int32.
            The counts of each of the control
            and treament groups of the left node, i.e. [N(T=i)...]

        alpha : float
            The weight used to balance different normalization parts.

        Returns
        -------
        norm_res : float
            Normalization factor.
        '''
        cdef N_TYPE_t[::1] cur_summary_n = cur_node_summary_n
        cdef N_TYPE_t[::1] left_summary_n = left_node_summary_n        
        cdef int n_class = cur_summary_n.shape[0]
        cdef int i = 0

        cdef P_TYPE_t norm_res = 0.0
        cdef P_TYPE_t n_c = cur_summary_n[0]
        cdef P_TYPE_t n_c_left = left_summary_n[0]
        cdef P_TYPE_t pt_a = 0.0, pt_a_i = 0.0, pc_a = 0.0, sum_n_t_left = 0.0, sum_n_t = 0.0

        for i in range(1, n_class):
            sum_n_t_left += left_summary_n[i]
            sum_n_t += cur_summary_n[i]

        pt_a = 1. * sum_n_t_left / (sum_n_t + 0.1)
        pc_a = 1. * n_c_left / (n_c + 0.1)

        if self.evaluationFunction == self.evaluate_IDDP:
            # Normalization Part 1
            norm_res += (entropyH(1. * sum_n_t / (sum_n_t + n_c), 1. * n_c / (sum_n_t + n_c)) * currentDivergence)
            norm_res += (1. * sum_n_t / (sum_n_t + n_c) * entropyH(pt_a))

        else:
            # Normalization Part 1
            norm_res += (alpha * entropyH(1. * sum_n_t / (sum_n_t + n_c), 1. * n_c / (sum_n_t + n_c)) * kl_divergence(pt_a, pc_a))
            # Normalization Part 2 & 3
            for i in range(1, n_class):
                pt_a_i = 1. * left_summary_n[i] / (cur_summary_n[i] + 0.1)
                norm_res += ((1 - alpha) * entropyH(1. * cur_summary_n[i] / (cur_summary_n[i] + n_c), 1. * n_c / (cur_summary_n[i] + n_c)) * kl_divergence(1. * pt_a_i, pc_a))
                norm_res += (1. * cur_summary_n[i] / (sum_n_t + n_c) * entropyH(pt_a_i))
        # Normalization Part 4
        norm_res += 1. * n_c / (sum_n_t + n_c) * entropyH(pc_a)

        # Normalization Part 5
        norm_res += 0.5
        return norm_res

    def tree_node_summary(self, treatment_idx, y, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        '''
        Tree node summary statistics.

        Args
        ----
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group t be split at a leaf node.
        n_reg :  int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : list of list
            The positive probabilities and sample sizes of each of the control and treatment groups
            in the parent node.

        Returns
        -------
        nodeSummary : list of list
            The positive probabilities and sample sizes of each of the control and treatment groups
            in the current node.
        '''
        # counts: [[N(Y=0, T=0), N(Y=1, T=0)], [N(Y=0, T=1), N(Y=1, T=1)], ...]
        counts = self.group_uniqueCounts(treatment_idx, y)

        # nodeSummary: [[P(Y=1|T=0), N(T=0)], [P(Y=1|T=1), N(T=1)], ...]
        nodeSummary = []
        # Iterate the control and treatment groups
        for i, count in enumerate(counts):
            n_pos = count[1]
            n = count[0] + n_pos
            if parentNodeSummary is None:
                p = n_pos / n if n > 0 else 0.
            elif n > min_samples_treatment:
                p = (n_pos + parentNodeSummary[i][0] * n_reg) / (n + n_reg)
            else:
                p = parentNodeSummary[i][0]

            nodeSummary.append([p, n])

        return nodeSummary

    @staticmethod
    def tree_node_summary_to_arr(np.ndarray[TR_TYPE_t, ndim=1] treatment_idx,
                                 np.ndarray[Y_TYPE_t, ndim=1] y,
                                 np.ndarray[P_TYPE_t, ndim=1] out_summary_p,
                                 np.ndarray[N_TYPE_t, ndim=1] out_summary_n,
                                 np.ndarray[N_TYPE_t, ndim=1] buf_count_arr,
                                 np.ndarray[P_TYPE_t, ndim=1] parentNodeSummary_p,
                                 int has_parent_summary,
                                 min_samples_treatment=10, n_reg=100
                                 ):
        '''
        Tree node summary statistics.
        Modified from tree_node_summary, to use different format for the summary.
        Instead of [[P(Y=1|T=0), N(T=0)], [P(Y=1|T=1), N(T=1)], ...],
        use two arrays [N(T=i)...] and [P(Y=1|T=i)...].

        Args
        ----
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
            Has type numpy.int8.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
            Has type numpy.int8.
        out_summary_p : array of shape [n_class]
            Has type numpy.double.
            To be filled with the positive probabilities of each of the control
            and treament groups of the current node.
        out_summary_n : array of shape [n_class]
            Has type numpy.int32.
            To be filled with the counts of each of the control
            and treament groups of the current node.
        buf_count_arr : array of shape [2*n_class]
            Has type numpy.int32.
            To be use as temporary buffer for group_uniqueCounts_to_arr.
        parentNodeSummary_p : array of shape [n_class]
            The positive probabilities of each of the control and treatment groups
            in the parent node.
        has_parent_summary : bool as int
            If True (non-zero), then parentNodeSummary_p is a valid parent node summary probabilities.
            If False (0), assume no parent node summary and parentNodeSummary_p is not touched.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group t be split at a leaf node.
        n_reg :  int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

        Returns
        -------
        No return values, but will modify out_summary_p and out_summary_n.
        '''
        # buf_count_arr: [N(Y=0, T=0), N(Y=1, T=0), N(Y=0, T=1), N(Y=1, T=1), ...]
        group_uniqueCounts_to_arr(treatment_idx, y, buf_count_arr)

        cdef int i = 0
        cdef int n_class = buf_count_arr.shape[0] / 2
        cdef int n = 0
        cdef int n_pos = 0
        cdef P_TYPE_t p = 0.0
        cdef int n_min_sams = min_samples_treatment
        cdef P_TYPE_t n_reg_p = n_reg

        # out_summary_p: [P(Y=1|T=i)...]
        # out_summary_n: [N(T=i) ... ]
        if has_parent_summary == 0:
            for i in range(n_class):
                n_pos = buf_count_arr[2*i + 1] # N(Y=1|T=i)
                n = buf_count_arr[2*i] + n_pos # N(Y=0|T=i) + N(Y=1|T=i) == N(T=i)
                p = (n_pos / <double> n) if n > 0 else 0.
                out_summary_n[i] = n
                out_summary_p[i] = p
        else:
            for i in range(n_class):
                n_pos = buf_count_arr[2*i + 1]
                n = buf_count_arr[2*i] + n_pos
                if n > n_min_sams:
                    p = (n_pos + parentNodeSummary_p[i] * n_reg_p) / (<double> n + n_reg_p)
                else:
                    p = parentNodeSummary_p[i]
                out_summary_n[i] = n
                out_summary_p[i] = p

    @staticmethod
    def tree_node_summary_from_counts(
            np.ndarray[N_TYPE_t, ndim=1] group_count_arr,
            np.ndarray[P_TYPE_t, ndim=1] out_summary_p,
            np.ndarray[N_TYPE_t, ndim=1] out_summary_n,
            np.ndarray[P_TYPE_t, ndim=1] parentNodeSummary_p,
            int has_parent_summary,
            min_samples_treatment=10, n_reg=100
    ):
        '''Tree node summary statistics.

        Modified from tree_node_summary_to_arr, to use different
        format for the summary and to calculate based on already
        calculated group counts.  Instead of [[P(Y=1|T=0), N(T=0)],
        [P(Y=1|T=1), N(T=1)], ...], use two arrays [N(T=i)...] and
        [P(Y=1|T=i)...].

        Args
        ----
        group_count_arr : array of shape [2*n_class]
            Has type numpy.int32.
            The grounp counts, where entry 2*i is N(Y=0, T=i),
            and entry 2*i+1 is N(Y=1, T=i).
        out_summary_p : array of shape [n_class]
            Has type numpy.double.
            To be filled with the positive probabilities of each of the control
            and treament groups of the current node.
        out_summary_n : array of shape [n_class]
            Has type numpy.int32.
            To be filled with the counts of each of the control
            and treament groups of the current node.
        parentNodeSummary_p : array of shape [n_class]
            The positive probabilities of each of the control and treatment groups
            in the parent node.
        has_parent_summary : bool as int
            If True (non-zero), then parentNodeSummary_p is a valid parent node summary probabilities.
            If False (0), assume no parent node summary and parentNodeSummary_p is not touched.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group t be split at a leaf node.
        n_reg :  int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

        Returns
        -------
        No return values, but will modify out_summary_p and out_summary_n.

        '''
        # group_count_arr: [N(Y=0, T=0), N(Y=1, T=0), N(Y=0, T=1), N(Y=1, T=1), ...]
        cdef int i = 0
        cdef int n_class = group_count_arr.shape[0] / 2
        cdef int n = 0
        cdef int n_pos = 0
        cdef P_TYPE_t p = 0.0
        cdef int n_min_sams = min_samples_treatment
        cdef P_TYPE_t n_reg_p = n_reg

        # out_summary_p: [P(Y=1|T=i)...]
        # out_summary_n: [N(T=i) ... ]
        if has_parent_summary == 0:
            for i in range(n_class):
                n_pos = group_count_arr[2*i + 1] # N(Y=1|T=i)
                n = group_count_arr[2*i] + n_pos # N(Y=0|T=i) + N(Y=1|T=i) == N(T=i)
                p = (n_pos / <double> n) if n > 0 else 0.
                out_summary_n[i] = n
                out_summary_p[i] = p
        else:
            for i in range(n_class):
                n_pos = group_count_arr[2*i + 1]
                n = group_count_arr[2*i] + n_pos
                if n > n_min_sams:
                    p = (n_pos + parentNodeSummary_p[i] * n_reg_p) / (<double> n + n_reg_p)
                else:
                    p = parentNodeSummary_p[i]
                out_summary_n[i] = n
                out_summary_p[i] = p

    def uplift_classification_results(self, treatment_idx, y):
        '''
        Classification probability for each treatment in the tree node.

        Args
        ----
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        res : list of list
            The positive probabilities P(Y = 1) of each of the control and treatment groups
        '''
        # counts: [[N(Y=0, T=0), N(Y=1, T=0)], [N(Y=0, T=1), N(Y=1, T=1)], ...]
        counts = self.group_uniqueCounts(treatment_idx, y)
        res = []
        for count in counts:
            n_pos = count[1]
            n = count[0] + n_pos
            p = n_pos / n if n > 0 else 0.
            res.append(p)
        return res

    def growDecisionTreeFrom(self, X, treatment_idx, y, X_val, treatment_val_idx, y_val,
                             early_stopping_eval_diff_scale=1, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary_p=None):
        '''
        Train the uplift decision tree.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group idx for each unit.
            The dtype should be numpy.int8.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        X_val : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to valid the uplift model.
        treatment_val_idx : array-like, shape = [num_samples]
            An array containing the validation treatment group idx for each unit.
        y_val : array-like, shape = [num_samples]
            An array containing the validation outcome of interest for each unit.
        max_depth: int, optional (default=10)
            The maximum depth of the tree.
        min_samples_leaf: int, optional (default=100)
            The minimum number of samples required to be split at a leaf node.
        depth : int, optional (default = 1)
            The current depth.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group to be split at a leaf node.
        n_reg: int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary_p : array-like, shape [n_class]
            Node summary probability statistics of the parent tree node.

        Returns
        -------
        object of DecisionTree class
        '''

        if len(X) == 0:
            return DecisionTree(classes_=self.classes_)

        assert treatment_idx.dtype == TR_TYPE
        assert y.dtype == Y_TYPE

        # some temporary buffers for node summaries
        cdef int n_class = self.n_class
        # buffers for group counts, right can be derived from total and left
        cdef np.ndarray[N_TYPE_t, ndim=1] left_count_arr = np.zeros(2 * self.n_class, dtype = N_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] right_count_arr = np.zeros(2 * self.n_class, dtype = N_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] total_count_arr = np.zeros(2 * self.n_class, dtype = N_TYPE)
        # for X_val if any, allocate if needed below
        cdef np.ndarray[N_TYPE_t, ndim=1] val_left_count_arr
        cdef np.ndarray[N_TYPE_t, ndim=1] val_right_count_arr
        cdef np.ndarray[N_TYPE_t, ndim=1] val_total_count_arr
        # buffers for node summary
        cdef np.ndarray[P_TYPE_t, ndim=1] cur_summary_p = np.zeros(self.n_class, dtype = P_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] cur_summary_n = np.zeros(self.n_class, dtype = N_TYPE)
        cdef np.ndarray[P_TYPE_t, ndim=1] left_summary_p = np.zeros(self.n_class, dtype = P_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] left_summary_n = np.zeros(self.n_class, dtype = N_TYPE)
        cdef np.ndarray[P_TYPE_t, ndim=1] right_summary_p = np.zeros(self.n_class, dtype = P_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] right_summary_n = np.zeros(self.n_class, dtype = N_TYPE)
        # for val left and right summary
        cdef np.ndarray[P_TYPE_t, ndim=1] val_left_summary_p = np.zeros(self.n_class, dtype = P_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] val_left_summary_n = np.zeros(self.n_class, dtype = N_TYPE)
        cdef np.ndarray[P_TYPE_t, ndim=1] val_right_summary_p = np.zeros(self.n_class, dtype = P_TYPE)
        cdef np.ndarray[N_TYPE_t, ndim=1] val_right_summary_n = np.zeros(self.n_class, dtype = N_TYPE)
        
        # dummy
        cdef int has_parent_summary = 0
        if parentNodeSummary_p is None:
            parent_summary_p = np.zeros(self.n_class, dtype = P_TYPE) # dummy for calling tree_node_summary_to_arr
            has_parent_summary = 0
        else:
            parent_summary_p = parentNodeSummary_p
            has_parent_summary = 1

        cdef int i = 0

        # preparation: fill in the total count, then for each
        # candidate split, we calculate the count for left branch, and
        # can derive count for right branch using the total count.

        # group_count_arr: [N(Y=0, T=0), N(Y=1, T=0), N(Y=0, T=1), N(Y=1, T=1), ...]
        group_uniqueCounts_to_arr(treatment_idx, y, total_count_arr)
        if X_val is not None:
            val_left_count_arr = np.zeros(2 * self.n_class, dtype = N_TYPE)
            val_right_count_arr = np.zeros(2 * self.n_class, dtype = N_TYPE)
            val_total_count_arr = np.zeros(2 * self.n_class, dtype = N_TYPE)
            group_uniqueCounts_to_arr(treatment_val_idx, y_val, val_total_count_arr)

        # Current node summary: [P(Y=1|T=i)...] and [N(T=i)...]
        self.tree_node_summary_from_counts(
            total_count_arr,
            cur_summary_p, cur_summary_n,
            parent_summary_p,
            has_parent_summary,
            min_samples_treatment=min_samples_treatment,
            n_reg=n_reg
            )

        # to reconstruct current node summary in list of list form, so
        # that the constructed tree follows previous format.

        # Current node summary: [[P(Y=1|T=i), N(T=i)]...]
        currentNodeSummary = []
        for i in range(n_class):
            currentNodeSummary.append([cur_summary_p[i], cur_summary_n[i]])
        #

        if self.evaluationFunction == self.evaluate_IT or self.evaluationFunction == self.evaluate_CIT:
            currentScore = 0
        else:
            currentScore = self.arr_eval_func(cur_summary_p, cur_summary_n)

        # Prune Stats:
        cdef P_TYPE_t maxAbsDiff = 0.0
        cdef P_TYPE_t maxDiff = -1.
        cdef int bestTreatment = 0       # treatment index for the control group, also used in returning the tree for this node
        cdef int suboptTreatment = 0     # treatment index for the control group
        cdef int maxDiffTreatment = 0    # treatment index for the control group, also used in returning the tree for this node
        maxDiffSign = 0 # also used in returning the tree for this node
        # adapted to new current node summary format
        cdef P_TYPE_t p_c = cur_summary_p[0]
        cdef N_TYPE_t n_c = cur_summary_n[0]
        cdef N_TYPE_t n_t = 0
        cdef int i_tr = 0
        cdef P_TYPE_t p_t = 0.0, diff = 0.0

        for i_tr in range(1, n_class):
            p_t = cur_summary_p[i_tr]
            # P(Y=1|T=t) - P(Y=1|T=0)
            diff = p_t - p_c
            if fabs(diff) >= maxAbsDiff:
                maxDiffTreatment = i_tr
                maxDiffSign = np.sign(diff)
                maxAbsDiff = fabs(diff)
            if diff >= maxDiff:
                maxDiff = diff
                suboptTreatment = i_tr
                if diff > 0:
                    bestTreatment = i_tr
        if maxDiff > 0:
            p_t = cur_summary_p[bestTreatment]
            n_t = cur_summary_n[bestTreatment]
        else:
            p_t = cur_summary_p[suboptTreatment]
            n_t = cur_summary_n[suboptTreatment]
        p_value = (1. - stats.norm.cdf(fabs(p_c - p_t) / sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))) * 2
        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestGainImp = 0.0
        bestAttribute = None
        # keep mostly scalar when finding best split, then get the structural value after finding the best split
        best_col = None
        best_value = None
        len_X = len(X)
        len_X_val = len(X_val) if X_val is not None else 0

        c_num_percentiles = [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]
        c_cat_percentiles = [10, 50, 90]

        # last column is the result/target column, 2nd to the last is the treatment group
        columnCount = X.shape[1]
        if (self.max_features and self.max_features > 0 and self.max_features <= columnCount):
            max_features = self.max_features
        else:
            max_features = columnCount

        for col in list(self.random_state_.choice(a=range(columnCount), size=max_features, replace=False)):
            columnValues = X[:, col]
            # unique values
            lsUnique = np.unique(columnValues)

            if np.issubdtype(lsUnique.dtype, np.number):
                is_split_by_gt = True
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, c_num_percentiles)
                else:
                    lspercentile = np.percentile(lsUnique, c_cat_percentiles)
                lsUnique = np.unique(lspercentile)
            else:
                # to split by equality check.
                is_split_by_gt = False

            for value in lsUnique:
                len_X_l = group_counts_by_divide(columnValues, value, is_split_by_gt, treatment_idx, y, left_count_arr)
                len_X_r = len_X - len_X_l

                # check the split validity on min_samples_leaf  372
                if (len_X_l < min_samples_leaf or len_X_r < min_samples_leaf):
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                p = float(len_X_l) / len_X

                # right branch group counts can be calculated from left branch counts and total counts
                for i in range(2 * n_class):
                    right_count_arr[i] = total_count_arr[i] - left_count_arr[i]

                # left and right node summary, into the temporary buffers {left,right}_summary_{p,n}
                self.tree_node_summary_from_counts(
                    left_count_arr,
                    left_summary_p, left_summary_n,
                    cur_summary_p,
                    1,
                    min_samples_treatment,
                    n_reg
                    )

                self.tree_node_summary_from_counts(
                    right_count_arr,
                    right_summary_p, right_summary_n,
                    cur_summary_p,
                    1,
                    min_samples_treatment,
                    n_reg
                    )

                if X_val is not None:
                    len_X_val_l = group_counts_by_divide(X_val[:, col], value, is_split_by_gt, treatment_val_idx, y_val, val_left_count_arr)

                    # right branch group counts can be calculated from left branch counts and total counts
                    for i in range(2 * n_class):
                        val_right_count_arr[i] = val_total_count_arr[i] - val_left_count_arr[i]

                    self.tree_node_summary_from_counts(
                        val_left_count_arr,
                        val_left_summary_p, val_left_summary_n,
                        cur_summary_p, # parentNodeSummary_p
                        1 # has_parent_summary
                    )

                    self.tree_node_summary_from_counts(
                        val_right_count_arr,
                        val_right_summary_p, val_right_summary_n,
                        cur_summary_p, # parentNodeSummary_p
                        1 # has_parent_summary
                    )

                    early_stopping_flag = False
                    for k in range(n_class):
                        if (abs(val_left_summary_p[k] - left_summary_p[k]) >
                                min(val_left_summary_p[k], left_summary_p[k])/early_stopping_eval_diff_scale or
                            abs(val_right_summary_p[k] - right_summary_p[k]) > 
                                min(val_right_summary_p[k], right_summary_p[k])/early_stopping_eval_diff_scale):
                            early_stopping_flag = True
                            break

                    if early_stopping_flag:
                        continue

                # check the split validity on min_samples_treatment
                node_mst = min(np.min(left_summary_n), np.min(right_summary_n))
                if node_mst < min_samples_treatment:
                    continue

                # evaluate the split
                if self.arr_eval_func == self.arr_evaluate_CTS:
                    leftScore1 = self.arr_eval_func(left_summary_p, left_summary_n)
                    rightScore2 = self.arr_eval_func(right_summary_p, right_summary_n)
                    gain = (currentScore - p * leftScore1 - (1 - p) * rightScore2)
                    gain_for_imp = (len_X * currentScore - len_X_l * leftScore1 - len_X_r * rightScore2)
                elif self.arr_eval_func == self.arr_evaluate_DDP:
                    leftScore1 = self.arr_eval_func(left_summary_p, left_summary_n)
                    rightScore2 = self.arr_eval_func(right_summary_p, right_summary_n)
                    gain = np.abs(leftScore1 - rightScore2)
                    gain_for_imp = np.abs(len_X_l * leftScore1 - len_X_r * rightScore2)
                elif self.arr_eval_func == self.arr_evaluate_IT:
                    gain = self.arr_eval_func(left_summary_p, left_summary_n, right_summary_p, right_summary_n)
                    gain_for_imp = gain * len_X
                elif self.arr_eval_func == self.arr_evaluate_CIT:
                    gain = self.arr_eval_func(cur_summary_p, cur_summary_n,
                                              left_summary_p, left_summary_n,
                                              right_summary_p, right_summary_n)
                    gain_for_imp = gain * len_X
                elif self.arr_eval_func == self.arr_evaluate_IDDP:
                    leftScore1 = self.arr_eval_func(left_summary_p, left_summary_n)
                    rightScore2 = self.arr_eval_func(right_summary_p, right_summary_n)
                    gain = np.abs(leftScore1 - rightScore2) - np.abs(currentScore)
                    gain_for_imp = (len_X_l * leftScore1 + len_X_r * rightScore2 - len_X * np.abs(currentScore))
                    if self.normalization:
                        # Normalize used divergence
                        currentDivergence = 2 * (gain + 1) / 3
                        norm_factor = self.arr_normI(cur_summary_n, left_summary_n, alpha=0.9, currentDivergence=currentDivergence)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                else:
                    leftScore1 = self.arr_eval_func(left_summary_p, left_summary_n)
                    rightScore2 = self.arr_eval_func(right_summary_p, right_summary_n)
                    gain = (p * leftScore1 + (1 - p) * rightScore2 - currentScore)
                    gain_for_imp = (len_X_l * leftScore1 + len_X_r * rightScore2 - len_X * currentScore)
                    if self.normalization:
                        norm_factor = self.arr_normI(cur_summary_n, left_summary_n, alpha=0.9)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor 
                if (gain > bestGain and len_X_l > min_samples_leaf and len_X_r > min_samples_leaf):
                    bestGain = gain
                    bestGainImp = gain_for_imp
                    best_col = col
                    best_value = value
        
        # after finding the best split col and value
        if best_col is not None:
            bestAttribute = (best_col, best_value)
            # re-calculate the divideSet
            X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, best_col, best_value)
            if X_val is not None:
                X_val_l, X_val_r, w_val_l, w_val_r, y_val_l, y_val_r = self.divideSet(X_val, treatment_val_idx, y_val, best_col, best_value)
                best_set_left = [X_l, w_l, y_l, X_val_l, w_val_l, y_val_l]
                best_set_right = [X_r, w_r, y_r, X_val_r, w_val_r, y_val_r]
            else:
                best_set_left = [X_l, w_l, y_l, None, None, None]
                best_set_right = [X_r, w_r, y_r, None, None, None]

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(X)}
        # Add treatment size
        dcY['group_size'] = ''
        for i, summary in enumerate(currentNodeSummary):
            dcY['group_size'] += ' ' + self.classes_[i] + ': ' + str(summary[1])
        dcY['upliftScore'] = [round(upliftScore[0], 4), round(upliftScore[1], 4)]
        dcY['matchScore'] = round(upliftScore[0], 4)

        if bestGain > 0 and depth < max_depth:
            self.feature_imp_dict[bestAttribute[0]] += bestGainImp
            trueBranch = self.growDecisionTreeFrom(
                *best_set_left, self.early_stopping_eval_diff_scale, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary_p=cur_summary_p
            )
            falseBranch = self.growDecisionTreeFrom(
                *best_set_right, self.early_stopping_eval_diff_scale, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary_p=cur_summary_p
            )

            return DecisionTree(
                classes_=self.classes_,
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(treatment_idx, y),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:
            if self.evaluationFunction == self.evaluate_CTS:
                return DecisionTree(
                    classes_=self.classes_,
                    results=self.uplift_classification_results(treatment_idx, y),
                    summary=dcY, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )
            else:
                return DecisionTree(
                    classes_=self.classes_,
                    results=self.uplift_classification_results(treatment_idx, y),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )

    @staticmethod
    def classify(observations, tree, dataMissing=False):
        '''
        Classifies (prediction) the observations according to the tree.

        Args
        ----
        observations : list of list
            The internal data format for the training data (combining X, Y, treatment).

        dataMissing: boolean, optional (default = False)
            An indicator for if data are missing or not.

        Returns
        -------
        tree.results, tree.upliftScore :
            The results in the leaf node.
        '''

        def classifyWithoutMissingData(observations, tree):
            '''
            Classifies (prediction) the observations according to the tree, assuming without missing data.

            Args
            ----
            observations : list of list
                The internal data format for the training data (combining X, Y, treatment).

            Returns
            -------
            tree.results, tree.upliftScore :
                The results in the leaf node.
            '''
            if tree.results is not None:  # leaf
                return tree.results, tree.upliftScore
            else:
                v = observations[tree.col]
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                else:
                    if v == tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
            return classifyWithoutMissingData(observations, branch)

        def classifyWithMissingData(observations, tree):
            '''
            Classifies (prediction) the observations according to the tree, assuming with missing data.

            Args
            ----
            observations : list of list
                The internal data format for the training data (combining X, Y, treatment).

            Returns
            -------
            tree.results, tree.upliftScore :
                The results in the leaf node.
            '''
            if tree.results is not None:  # leaf
                return tree.results
            else:
                v = observations[tree.col]
                if v is None:
                    tr = classifyWithMissingData(observations, tree.trueBranch)
                    fr = classifyWithMissingData(observations, tree.falseBranch)
                    tcount = sum(tr.values())
                    fcount = sum(fr.values())
                    tw = float(tcount) / (tcount + fcount)
                    fw = float(fcount) / (tcount + fcount)

                    # Problem description: http://blog.ludovf.net/python-collections-defaultdict/
                    result = defaultdict(int)
                    for k, v in tr.items():
                        result[k] += v * tw
                    for k, v in fr.items():
                        result[k] += v * fw
                    return dict(result)
                else:
                    branch = None
                    if isinstance(v, int) or isinstance(v, float):
                        if v >= tree.value:
                            branch = tree.trueBranch
                        else:
                            branch = tree.falseBranch
                    else:
                        if v == tree.value:
                            branch = tree.trueBranch
                        else:
                            branch = tree.falseBranch
                return classifyWithMissingData(observations, branch)

        # function body
        if dataMissing:
            return classifyWithMissingData(observations, tree)
        else:
            return classifyWithoutMissingData(observations, tree)


# Uplift Random Forests
class UpliftRandomForestClassifier:
    """ Uplift Random Forest for Classification Task.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the uplift random forest.

    evaluationFunction : string
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS', 'DDP', 'IT', 'CIT', 'IDDP'.

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    random_state: int, RandomState instance or None (default=None)
        A random seed or `np.random.RandomState` to control randomness in building the trees and forest.

    max_depth: int, optional (default=5)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=10)
        The regularization parameter defined in Rzepakowski et al. 2012, the
        weight (in terms of sample size) of the parent node influence on the
        child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

    early_stopping_eval_diff_scale: float, optional (default=1)
        If train and valid uplift score diff bigger than 
        min(train_uplift_score,valid_uplift_score)/early_stopping_eval_diff_scale, stop.

    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups)

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012,
        correcting for tests with large number of splits and imbalanced
        treatment and control splits

    honesty: bool (default=False)
     True if the honest approach based on "Athey, S., & Imbens, G. (2016). Recursive partitioning for
     heterogeneous causal effects." shall be used.

    estimation_sample_size: float (default=0.5)
         Sample size for estimating the CATE score in the leaves if honesty == True.

    n_jobs: int, optional (default=-1)
        The parallelization parameter to define how many parallel jobs need to be created.
        This is passed on to joblib library for parallelizing uplift-tree creation and prediction.

    joblib_prefer: str, optional (default="threads")
        The preferred backend for joblib (passed as `prefer` to joblib.Parallel). See the joblib
        documentation for valid values.

    Outputs
    ----------
    df_res: pandas dataframe
        A user-level results dataframe containing the estimated individual treatment effect.
    """
    def __init__(self,
                 control_name,
                 n_estimators=10,
                 max_features=10,
                 random_state=None,
                 max_depth=5,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 early_stopping_eval_diff_scale=1,
                 evaluationFunction='KL',
                 normalization=True,
                 honesty=False,
                 estimation_sample_size=0.5,
                 n_jobs=-1,
                 joblib_prefer: str = "threads"):

        """
        Initialize the UpliftRandomForestClassifier class.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.early_stopping_eval_diff_scale = early_stopping_eval_diff_scale
        self.evaluationFunction = evaluationFunction
        self.control_name = control_name
        self.normalization = normalization
        self.honesty = honesty
        self.n_jobs = n_jobs
        self.joblib_prefer = joblib_prefer

        assert control_name is not None and isinstance(control_name, str), \
            f"control_group should be string but {control_name} is passed"
        self.control_name = control_name
        self.classes_ = [control_name]
        self.n_class = 1

        if self.n_jobs == -1:
            self.n_jobs = mp.cpu_count()

    def fit(self, X, treatment, y, X_val=None, treatment_val=None, y_val=None):
        """
        Fit the UpliftRandomForestClassifier.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.

        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        X_val : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to valid the uplift model.

        treatment_val : array-like, shape = [num_samples]
            An array containing the validation treatment group for each unit.

        y_val : array-like, shape = [num_samples]
            An array containing the validation outcome of interest for each unit.
        """
        random_state = check_random_state(self.random_state)

        # Create forest
        self.uplift_forest = [
            UpliftTreeClassifier(
                max_features=self.max_features, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg,
                early_stopping_eval_diff_scale=self.early_stopping_eval_diff_scale,
                evaluationFunction=self.evaluationFunction,
                control_name=self.control_name,
                normalization=self.normalization,
                honesty=self.honesty,
                random_state=random_state.randint(MAX_INT))
            for _ in range(self.n_estimators)
        ]

        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_groups = sorted([x for x in list(set(treatment)) if x != self.control_name])
        self.classes_ = [self.control_name]
        for tr in treatment_groups:
            self.classes_.append(tr)
        self.n_class = len(self.classes_)

        self.uplift_forest = (
            Parallel(n_jobs=self.n_jobs, prefer=self.joblib_prefer)
            (delayed(self.bootstrap)(X, treatment, y, X_val, treatment_val, y_val, tree) for tree in self.uplift_forest)
        )

        all_importances = [tree.feature_importances_ for tree in self.uplift_forest]
        self.feature_importances_ = np.mean(all_importances, axis=0)
        self.feature_importances_ /= self.feature_importances_.sum()  # normalize to add to 1

    @staticmethod
    def bootstrap(X, treatment, y, X_val, treatment_val, y_val, tree):
        random_state = check_random_state(tree.random_state)
        bt_index = random_state.choice(len(X), len(X))
        x_train_bt = X[bt_index]
        y_train_bt = y[bt_index]
        treatment_train_bt = treatment[bt_index]

        if X_val is None:
            tree.fit(X=x_train_bt, treatment=treatment_train_bt, y=y_train_bt)
        else:
            bt_val_index = random_state.choice(len(X_val), len(X_val))
            x_val_bt = X_val[bt_val_index]
            y_val_bt = y_val[bt_val_index]
            treatment_val_bt = treatment_val[bt_val_index]
    
            tree.fit(X=x_train_bt, treatment=treatment_train_bt, y=y_train_bt, X_val=x_val_bt, treatment_val=treatment_val_bt, y_val=y_val_bt)
        return tree

    @ignore_warnings(category=FutureWarning)
    def predict(self, X, full_output=False):
        '''
        Returns the recommended treatment group and predicted optimal
        probability conditional on using the recommended treatment group.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        full_output : bool, optional (default=False)
            Whether the UpliftTree algorithm returns upliftScores, pred_nodes
            alongside the recommended treatment group and p_hat in the treatment group.

        Returns
        -------
        y_pred_list : ndarray, shape = (num_samples, num_treatments])
            An ndarray containing the predicted treatment effect of each treatment group for each sample

        df_res : DataFrame, shape = [num_samples, (num_treatments * 2 + 3)]
            If `full_output` is `True`, a DataFrame containing the predicted outcome of each treatment and
            control group, the treatment effect of each treatment group, the treatment group with the
            highest treatment effect, and the maximum treatment effect for each sample.

        '''
        # Make predictions with all trees and take the average

        if self.n_jobs != 1:
            y_pred_ensemble = sum(
                Parallel(n_jobs=self.n_jobs, prefer=self.joblib_prefer)
                (delayed(tree.predict)(X=X) for tree in self.uplift_forest)
            ) / len(self.uplift_forest)
        else:
            y_pred_ensemble = sum([tree.predict(X=X) for tree in self.uplift_forest]) / len(self.uplift_forest)

        # Summarize results into dataframe
        df_res = pd.DataFrame(y_pred_ensemble, columns=self.classes_)
        df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)

        # Calculate delta
        delta_cols = [f'delta_{treatment_group}' for treatment_group in self.classes_[1:]]
        for i_tr in range(1, self.n_class):
            treatment_group = self.classes_[i_tr]
            df_res[f'delta_{treatment_group}'] = df_res[treatment_group] - df_res[self.control_name]

        df_res['max_delta'] = df_res[delta_cols].max(axis=1)

        if full_output:
            return df_res
        else:
            return df_res[delta_cols].values
