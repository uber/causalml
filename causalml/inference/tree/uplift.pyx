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


MAX_INT = np.iinfo(np.int32).max

logger = logging.getLogger("causalml")

cdef extern from "math.h":
    double log(double x) nogil


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
                 min_samples_treatment=10, n_reg=100, evaluationFunction='KL',
                 normalization=True, honesty=False, estimation_sample_size=0.5, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.max_features = max_features

        assert evaluationFunction in ['KL', 'ED', 'Chi', 'CTS', 'DDP', 'IT', 'CIT', 'IDDP'], \
            f"evaluationFunction should be either 'KL', 'ED', 'Chi', 'CTS', 'DDP', 'IT', 'CIT', or 'IDDP' but {evaluationFunction} is passed"

        if evaluationFunction == 'KL':
            self.evaluationFunction = self.evaluate_KL
        elif evaluationFunction == 'ED':
            self.evaluationFunction = self.evaluate_ED
        elif evaluationFunction == 'Chi':
            self.evaluationFunction = self.evaluate_Chi
        elif evaluationFunction == 'DDP':
            self.evaluationFunction = self.evaluate_DDP
        elif evaluationFunction == 'IT':
            self.evaluationFunction = self.evaluate_IT
        elif evaluationFunction == 'CIT':
            self.evaluationFunction = self.evaluate_CIT
        elif evaluationFunction == 'IDDP':
            self.evaluationFunction = self.evaluate_IDDP
        elif evaluationFunction == 'CTS':
            self.evaluationFunction = self.evaluate_CTS
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


    def fit(self, X, treatment, y):
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
        treatment = np.asarray(treatment)
        assert len(y) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_groups = sorted([x for x in list(set(treatment)) if x != self.control_name])
        self.classes_ = [self.control_name]
        treatment_idx = np.zeros_like(treatment, dtype=int)
        for i, tr in enumerate(treatment_groups, 1):
            self.classes_.append(tr)
            treatment_idx[treatment == tr] = i
        self.n_class = len(self.classes_)

        self.feature_imp_dict = defaultdict(float)

        if (self.n_class > 2) and (self.evaluationFunction in [self.evaluate_DDP, self.evaluate_IDDP, self.evaluate_IT, self.evaluate_CIT]):
            raise ValueError("The DDP, IDDP, IT, and CIT approach can only cope with two class problems, that is two different treatment "
                             "options (e.g., control vs treatment). Please select another approach or only use a "
                             "dataset which employs two treatment options.")

        if self.honesty:
            try:
                X, X_est, treatment_idx, treatment_idx_est, y, y_est = train_test_split(X, treatment_idx, y, stratify=[treatment_idx, y], test_size=self.estimation_sample_size,
                                                                                        shuffle=True, random_state=self.random_state)
            except ValueError:
                logger.warning(f"Stratified sampling failed. Falling back to random sampling.")
                X, X_est, treatment_idx, treatment_idx_est, y, y_est = train_test_split(X, treatment_idx, y, test_size=self.estimation_sample_size, shuffle=True,
                                                                                        random_state=self.random_state)

        self.fitted_uplift_tree = self.growDecisionTreeFrom(
            X, treatment_idx, y,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            depth=1, min_samples_treatment=self.min_samples_treatment,
            n_reg=self.n_reg, parentNodeSummary=None
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

        self.modifyEstimation(X_est, T_est, Y_est, self.fitted_uplift_tree)

    def modifyEstimation(self, X_est, t_est, y_est, tree):
        """ Modifies the leafs of the current decision tree to only contain unbiased estimates.
        Applies the honest approach based on "Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects."
        Args
        ----
        X_est : ndarray, shape = [num_samples, num_features]
           An ndarray of the covariates used to calculate the unbiased estimates in the leafs of the decision tree.
        T_est : array-like, shape = [num_samples]
           An array containing the treatment group for each unit.
        Y_est : array-like, shape = [num_samples]
           An array containing the outcome of interest for each unit.
        tree : object
            object of DecisionTree class - the current decision tree that shall be modified
        """

        # Divide sets for child nodes
        if tree.trueBranch or tree.falseBranch:
            X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X_est, t_est, y_est, tree.col, tree.value)

            # recursive call for each branch
            if tree.trueBranch is not None:
                self.modifyEstimation(X_l, w_l, y_l, tree.trueBranch)
            if tree.falseBranch is not None:
                self.modifyEstimation(X_r, w_r, y_r, tree.falseBranch)

        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(t_est, y_est)

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
    def evaluate_CTS(nodeSummary):
        '''
        Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary() method.

        Returns
        -------
        d_res : Chi-Square
        '''
        return -max([stat[0] for stat in nodeSummary])

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

    def growDecisionTreeFrom(self, X, treatment_idx, y, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None):
        '''
        Train the uplift decision tree.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group idx for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
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
        parentNodeSummary : dictionary, optional (default = None)
            Node summary statistics of the parent tree node.

        Returns
        -------
        object of DecisionTree class
        '''

        if len(X) == 0:
            return DecisionTree(classes_=self.classes_)

        # Current node summary: [P(Y=1|T), N(T)]
        currentNodeSummary = self.tree_node_summary(treatment_idx, y,
                                                    min_samples_treatment=min_samples_treatment,
                                                    n_reg=n_reg,
                                                    parentNodeSummary=parentNodeSummary)

        if self.evaluationFunction == self.evaluate_IT or self.evaluationFunction == self.evaluate_CIT:
            currentScore = 0
        else:
            currentScore = self.evaluationFunction(currentNodeSummary)

        # Prune Stats
        maxAbsDiff = 0
        maxDiff = -1.
        bestTreatment = 0       # treatment index for the control group
        suboptTreatment = 0     # treatment index for the control group
        maxDiffTreatment = 0    # treatment index for the control group
        maxDiffSign = 0
        p_c, n_c = currentNodeSummary[0]
        for i_tr in range(1, self.n_class):
            p_t, _ = currentNodeSummary[i_tr]
            # P(Y|T=t) - P(Y|T=0)
            diff = p_t - p_c
            if abs(diff) >= maxAbsDiff:
                maxDiffTreatment = i_tr
                maxDiffSign = np.sign(diff)
                maxAbsDiff = abs(diff)
            if diff >= maxDiff:
                maxDiff = diff
                suboptTreatment = i_tr
                if diff > 0:
                    bestTreatment = i_tr
        if maxDiff > 0:
            p_t = currentNodeSummary[bestTreatment][0]
            n_t = currentNodeSummary[bestTreatment][1]
        else:
            p_t = currentNodeSummary[suboptTreatment][0]
            n_t = currentNodeSummary[suboptTreatment][1]

        p_value = (1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))) * 2
        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestGainImp = 0.0
        bestAttribute = None

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
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = np.unique(lspercentile)

            for value in lsUnique:
                X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, col, value)
                # check the split validity on min_samples_leaf  372
                if (len(X_l) < min_samples_leaf or len(X_r) < min_samples_leaf):
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                p = float(len(X_l)) / len(X)
                leftNodeSummary = self.tree_node_summary(w_l, y_l,
                                                         min_samples_treatment=min_samples_treatment,
                                                         n_reg=n_reg,
                                                         parentNodeSummary=currentNodeSummary)

                rightNodeSummary = self.tree_node_summary(w_r, y_r,
                                                          min_samples_treatment=min_samples_treatment,
                                                          n_reg=n_reg,
                                                          parentNodeSummary=currentNodeSummary)

                # check the split validity on min_samples_treatment
                assert len(leftNodeSummary) == len(rightNodeSummary)

                node_mst = min([stat[1] for stat in leftNodeSummary + rightNodeSummary])
                if node_mst < min_samples_treatment:
                    continue

                # evaluate the split
                if self.evaluationFunction == self.evaluate_CTS:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = (currentScore - p * leftScore1 - (1 - p) * rightScore2)
                    gain_for_imp = (len(X) * currentScore - len(X_l) * leftScore1 - len(X_r) * rightScore2)
                elif self.evaluationFunction == self.evaluate_DDP:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = np.abs(leftScore1 - rightScore2)
                    gain_for_imp = np.abs(len(X_l) * leftScore1 - len(X_r) * rightScore2)
                elif self.evaluationFunction == self.evaluate_IT:
                    gain = self.evaluationFunction(leftNodeSummary, rightNodeSummary, w_l, w_r)
                    gain_for_imp = gain * len(X)
                elif self.evaluationFunction == self.evaluate_CIT:
                    gain = self.evaluationFunction(currentNodeSummary, leftNodeSummary, rightNodeSummary, y_l, y_r, w_l, w_r, y, treatment_idx)
                    gain_for_imp = gain * len(X)
                elif self.evaluationFunction == self.evaluate_IDDP:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = np.abs(leftScore1 - rightScore2) - np.abs(currentScore)
                    gain_for_imp = (len(X_l) * leftScore1 + len(X_r) * rightScore2 - len(X) * np.abs(currentScore))
                    if self.normalization:
                        # Normalize used divergence
                        currentDivergence = 2 * (gain + 1) / 3
                        n_c = currentNodeSummary[0][1]
                        n_c_left = leftNodeSummary[0][1]
                        n_t = [tr[1] for tr in currentNodeSummary[1:]]
                        n_t_left = [tr[1] for tr in leftNodeSummary[1:]]
                        norm_factor = self.normI(n_c, n_c_left, n_t, n_t_left, alpha=0.9, currentDivergence=currentDivergence)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                else:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = (p * leftScore1 + (1 - p) * rightScore2 - currentScore)
                    gain_for_imp = (len(X_l) * leftScore1 + len(X_r) * rightScore2 - len(X) * currentScore)
                    if self.normalization:
                        n_c = currentNodeSummary[0][1]
                        n_c_left = leftNodeSummary[0][1]
                        n_t = [tr[1] for tr in currentNodeSummary[1:]]
                        n_t_left = [tr[1] for tr in leftNodeSummary[1:]]

                        norm_factor = self.normI(n_c, n_c_left, n_t, n_t_left, alpha=0.9)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                if (gain > bestGain and len(X_l) > min_samples_leaf and len(X_r) > min_samples_leaf):
                    bestGain = gain
                    bestGainImp = gain_for_imp
                    bestAttribute = (col, value)
                    best_set_left = [X_l, w_l, y_l]
                    best_set_right = [X_r, w_r, y_r]

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
                *best_set_left, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )
            falseBranch = self.growDecisionTreeFrom(
                *best_set_right, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
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

    def fit(self, X, treatment, y):
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
        """
        random_state = check_random_state(self.random_state)

        # Create forest
        self.uplift_forest = [
            UpliftTreeClassifier(
                max_features=self.max_features, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg,
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
            (delayed(self.bootstrap)(X, treatment, y, tree) for tree in self.uplift_forest)
        )

        all_importances = [tree.feature_importances_ for tree in self.uplift_forest]
        self.feature_importances_ = np.mean(all_importances, axis=0)
        self.feature_importances_ /= self.feature_importances_.sum()  # normalize to add to 1

    @staticmethod
    def bootstrap(X, treatment, y, tree):
        random_state = check_random_state(tree.random_state)
        bt_index = random_state.choice(len(X), len(X))
        x_train_bt = X[bt_index]
        y_train_bt = y[bt_index]
        treatment_train_bt = treatment[bt_index]
        tree.fit(X=x_train_bt, treatment=treatment_train_bt, y=y_train_bt)
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
