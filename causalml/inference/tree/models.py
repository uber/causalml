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

from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
from packaging import version
import pandas as pd
import scipy.stats as stats
import sklearn
if version.parse(sklearn.__version__) >= version.parse('0.22.0'):
    from sklearn.utils._testing import ignore_warnings
else:
    from sklearn.utils.testing import ignore_warnings


class DecisionTree:
    """ Tree Node Class

    Tree node class to contain all the statistics of the tree node.

    Parameters
    ----------

    col : int, optional (default = -1)
        The column index for splitting the tree node to children nodes.

    value : float, optional (default = None)
        The value of the feature column to split the tree node to children nodes.

    trueBranch : object of DecisionTree
        The true branch tree node (feature > value).

    falseBranch : object of DecisionTree
        The false branch tree node (feature > value).

    results : dictionary
        The classification probability Pr(1) for each experiment group in the tree node.

    summary : dictionary
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.

    maxDiffTreatment : string
        The treatment name generating the maximum difference between treatment and control group.

    maxDiffSign : float
        The sign of the maximum difference (1. or -1.).

    nodeSummary : dictionary
        Summary statistics of the tree nodes {treatment: [y_mean, n]}, where y_mean stands for the target metric mean
        and n is the sample size.

    backupResults : dictionary
        The conversion probabilities in each treatment in the parent node {treatment: y_mean}. The parent node
        information is served as a backup for the children node, in case no valid statistics can be calculated from the
        children node, the parent node information will be used in certain cases.

    bestTreatment : string
        The treatment name providing the best uplift (treatment effect).

    upliftScore : list
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maximum treatment effect, and
        p_value stands for the p_value of the treatment effect.

    matchScore : float
        The uplift score by filling a trained tree with validation dataset or testing dataset.

    """

    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None,
                 results=None, summary=None, maxDiffTreatment=None,
                 maxDiffSign=1., nodeSummary=None, backupResults=None,
                 bestTreatment=None, upliftScore=None, matchScore=None):
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
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

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
        The name of the control group (other experiment groups will be regarded as treatment groups)

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012, correcting for tests with large number of splits
        and imbalanced treatment and control splits

    """
    def __init__(self, max_features=None, max_depth=3, min_samples_leaf=100,
                 min_samples_treatment=10, n_reg=100, evaluationFunction='KL',
                 control_name=None, normalization=True):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.max_features = max_features
        if evaluationFunction == 'KL':
            self.evaluationFunction = self.evaluate_KL
        elif evaluationFunction == 'ED':
            self.evaluationFunction = self.evaluate_ED
        elif evaluationFunction == 'Chi':
            self.evaluationFunction = self.evaluate_Chi
        else:
            self.evaluationFunction = self.evaluate_CTS
        self.fitted_uplift_tree = None
        self.control_name = control_name
        self.normalization = normalization

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
        assert len(X) == len(y) and len(X) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        self.treatment_group = list(set(treatment))
        self.feature_imp_dict = defaultdict(float)

        self.fitted_uplift_tree = self.growDecisionTreeFrom(
            X, treatment, y, evaluationFunction=self.evaluationFunction,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            depth=1, min_samples_treatment=self.min_samples_treatment,
            n_reg=self.n_reg, parentNodeSummary=None
        )

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
        assert len(X) == len(y) and len(X) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        self.pruneTree(X, treatment, y,
                       tree=self.fitted_uplift_tree,
                       rule=rule,
                       minGain=minGain,
                       evaluationFunction=self.evaluationFunction,
                       notify=False,
                       n_reg=self.n_reg,
                       parentNodeSummary=None)
        return self

    def pruneTree(self, X, treatment, y, tree, rule='maxAbsDiff', minGain=0.,
                  evaluationFunction=None, notify=False, n_reg=0,
                  parentNodeSummary=None):
        """Prune one single tree node in the uplift model.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        rule : string, optional (default = 'maxAbsDiff')
            The prune rules. Supported values are 'maxAbsDiff' for optimizing the maximum absolute difference, and
            'bestUplift' for optimizing the node-size weighted treatment effect.
        minGain : float, optional (default = 0.)
            The minimum gain required to make a tree node split. The children tree branches are trimmed if the actual
            split gain is less than the minimum gain.
        evaluationFunction : string, optional (default = None)
            Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.
        notify: bool, optional (default = False)
        n_reg: int, optional (default=0)
            The regularization parameter defined in Rzepakowski et al. 2012, the weight (in terms of sample size) of the
        parent node influence on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : dictionary, optional (default = None)
            Node summary statistics of the parent tree node.
        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(
            treatment, y, min_samples_treatment=self.min_samples_treatment,
            n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment, y, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch.results is None:
            self.pruneTree(X_l, w_l, y_l, tree.trueBranch, rule, minGain,
                           evaluationFunction, notify, n_reg,
                           parentNodeSummary=currentNodeSummary)
        if tree.falseBranch.results is None:
            self.pruneTree(X_r, w_r, y_r, tree.falseBranch, rule, minGain,
                           evaluationFunction, notify, n_reg,
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
        assert len(X) == len(y) and len(X) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        self.fillTree(X, treatment, y, tree=self.fitted_uplift_tree)
        return self

    def fillTree(self, X, treatment, y, tree):
        """ Fill the data into an existing tree.
        This is a lower-level function to execute on the tree filling task.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        tree : object
            object of DecisionTree class

        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(treatment, y,
                                                    min_samples_treatment=0,
                                                    n_reg=0,
                                                    parentNodeSummary=None)
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment, y, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch is not None:
            self.fillTree(X_l, w_l, y_l, tree.trueBranch)
        if tree.falseBranch is not None:
            self.fillTree(X_r, w_r, y_r, tree.falseBranch)

        # Update Information

        # matchScore
        matchScore = (currentNodeSummary[tree.bestTreatment][0] - currentNodeSummary[self.control_name][0])
        tree.matchScore = round(matchScore, 4)
        tree.summary['matchScore'] = round(matchScore, 4)

        # Samples, Group_size
        tree.summary['samples'] = len(y)
        tree.summary['group_size'] = ''
        for treatment_group in currentNodeSummary:
            tree.summary['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(treatment, y)
        return self

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
        df_res : DataFrame, shape = [num_samples, (num_treatments + 1)]
            A DataFrame containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.

        '''

        p_hat_optimal = []
        treatment_optimal = []
        pred_nodes = {}
        upliftScores = []
        for xi in range(len(X)):
            pred_leaf, upliftScore = self.classify(X[xi], self.fitted_uplift_tree, dataMissing=False)
            # Predict under uplift optimal treatment
            opt_treat = max(pred_leaf, key=pred_leaf.get)
            p_hat_optimal.append(pred_leaf[opt_treat])
            treatment_optimal.append(opt_treat)
            if full_output:
                if xi == 0:
                    for key_i in pred_leaf:
                        pred_nodes[key_i] = [pred_leaf[key_i]]
                else:
                    for key_i in pred_leaf:
                        pred_nodes[key_i].append(pred_leaf[key_i])
                upliftScores.append(upliftScore)
        if full_output:
            return treatment_optimal, p_hat_optimal, upliftScores, pred_nodes
        else:
            return treatment_optimal, p_hat_optimal

    @staticmethod
    def divideSet(X, treatment, y, column, value):
        '''
        Tree node split.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
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
        if isinstance(value, int) or isinstance(value, float):
            filt = X[:, column] >= value
        else:  # for strings
            filt = X[:, column] == value

        return X[filt], X[~filt], treatment[filt], treatment[~filt], y[filt], y[~filt]

    def group_uniqueCounts(self, treatment, y):
        '''
        Count sample size by experiment group.

        Args
        ----
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        results : dictionary
                The control and treatment sample size.
        '''
        results = {}
        for t in self.treatment_group:
            filt = treatment == t
            n_t = y[filt].sum()
            results[t] = (filt.sum() - n_t, n_t)

        return results

    @staticmethod
    def kl_divergence(pk, qk):
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

        eps = 1e-6
        qk = np.clip(qk, eps, 1 - eps)

        if pk == 0:
            S = -np.log(1 - qk)
        elif pk == 1:
            S = -np.log(qk)
        else:
            S = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))

        return S

    def evaluate_KL(self, nodeSummary, control_name):
        '''
        Calculate KL Divergence as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary()
            method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : KL Divergence
        '''
        if control_name not in nodeSummary:
            return 0
        pc = nodeSummary[control_name][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                d_res += self.kl_divergence(nodeSummary[treatment_group][0], pc)
        return d_res

    @staticmethod
    def evaluate_ED(nodeSummary, control_name):
        '''
        Calculate Euclidean Distance as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary()
            method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : Euclidean Distance
        '''
        if control_name not in nodeSummary:
            return 0
        pc = nodeSummary[control_name][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                d_res += 2*(nodeSummary[treatment_group][0] - pc)**2
        return d_res

    @staticmethod
    def evaluate_Chi(nodeSummary, control_name):
        '''
        Calculate Chi-Square statistic as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : Chi-Square
        '''
        if control_name not in nodeSummary:
            return 0
        pc = nodeSummary[control_name][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                d_res += ((nodeSummary[treatment_group][0] - pc) ** 2 / max(0.1 ** 6, pc)
                          + (nodeSummary[treatment_group][0] - pc) ** 2 / max(0.1 ** 6, 1 - pc))
        return d_res

    @staticmethod
    def evaluate_CTS(currentNodeSummary):
        '''
        Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.

        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : Chi-Square
        '''
        mu = 0.0
        # iterate treatment group
        for r in currentNodeSummary:
            mu = max(mu, currentNodeSummary[r][0])
        return -mu

    @staticmethod
    def entropyH(p, q=None):
        '''
        Entropy

        Entropy calculation for normalization.

        Args
        ----
        p : float
            The probability used in the entropy calculation.

        q : float, optional, (default = None)
            The second probability used in the entropy calculation.

        Returns
        -------
        entropy : float
        '''
        if q is None and p > 0:
            return -p * np.log(p)
        elif q > 0:
            return -p * np.log(q)
        else:
            return 0

    def normI(self, currentNodeSummary, leftNodeSummary, rightNodeSummary, control_name, alpha=0.9):
        '''
        Normalization factor.

        Args
        ----
        currentNodeSummary : dictionary
            The summary statistics of the current tree node.

        leftNodeSummary : dictionary
            The summary statistics of the left tree node.

        rightNodeSummary : dictionary
            The summary statistics of the right tree node.

        control_name : string
            The control group name.

        alpha : float
            The weight used to balance different normalization parts.

        Returns
        -------
        norm_res : float
            Normalization factor.
        '''
        norm_res = 0
        # n_t, n_c: sample size for all treatment, and control
        # pt_a, pc_a: % of treatment is in left node, % of control is in left node
        n_c = currentNodeSummary[control_name][1]
        n_c_left = leftNodeSummary[control_name][1]
        n_t = []
        n_t_left = []
        for treatment_group in currentNodeSummary:
            if treatment_group != control_name:
                n_t.append(currentNodeSummary[treatment_group][1])
                if treatment_group in leftNodeSummary:
                    n_t_left.append(leftNodeSummary[treatment_group][1])
                else:
                    n_t_left.append(0)
        pt_a = 1. * np.sum(n_t_left) / (np.sum(n_t) + 0.1)
        pc_a = 1. * n_c_left / (n_c + 0.1)
        # Normalization Part 1
        norm_res += (
            alpha * self.entropyH(1. * np.sum(n_t) / (np.sum(n_t) + n_c), 1. * n_c / (np.sum(n_t) + n_c))
            * self.kl_divergence(pt_a, pc_a)
        )
        # Normalization Part 2 & 3
        for i in range(len(n_t)):
            pt_a_i = 1. * n_t_left[i] / (n_t[i] + 0.1)
            norm_res += (
                (1 - alpha) * self.entropyH(1. * n_t[i] / (n_t[i] + n_c), 1. * n_c / (n_t[i] + n_c))
                * self.kl_divergence(1. * pt_a_i, pc_a)
            )
            norm_res += (1. * n_t[i] / (np.sum(n_t) + n_c) * self.entropyH(pt_a_i))
        # Normalization Part 4
        norm_res += 1. * n_c/(np.sum(n_t) + n_c) * self.entropyH(pc_a)

        # Normalization Part 5
        norm_res += 0.5
        return norm_res

    def tree_node_summary(self, treatment, y, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        '''
        Tree node summary statistics.

        Args
        ----
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group t be split at a leaf node.
        n_reg :  int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : dictionary
            Node summary statistics of the parent tree node.

        Returns
        -------
        nodeSummary : dictionary
            The node summary of the current tree node.
        '''
        # returns {treatment_group: p(1)}
        results = self.group_uniqueCounts(treatment, y)
        # node Summary: {treatment_group: [p(1), size]}
        nodeSummary = {}
        # iterate treatment group
        for r in results:
            n1 = results[r][1]
            ntot = results[r][0] + n1
            if parentNodeSummary is None:
                y_mean = n1 / ntot
            elif ntot > min_samples_treatment:
                y_mean = (n1 + parentNodeSummary[r][0] * n_reg) / (ntot + n_reg)
            else:
                y_mean = parentNodeSummary[r][0]

            nodeSummary[r] = [y_mean, ntot]

        return nodeSummary

    def uplift_classification_results(self, treatment, y):
        '''
        Classification probability for each treatment in the tree node.

        Args
        ----
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        res : dictionary
            The probability of 1 in each treatment in the tree node.
        '''
        results = self.group_uniqueCounts(treatment, y)
        res = {}
        for r in results:
            p = float(results[r][1]) / (results[r][0] + results[r][1])
            res[r] = round(p, 6)
        return res

    def growDecisionTreeFrom(self, X, treatment, y, evaluationFunction, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None):
        '''
        Train the uplift decision tree.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        evaluationFunction : string
            Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.
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
            return DecisionTree()

        # Current Node Info and Summary
        currentNodeSummary = self.tree_node_summary(treatment, y,
                                                    min_samples_treatment=min_samples_treatment,
                                                    n_reg=n_reg,
                                                    parentNodeSummary=parentNodeSummary)
        if evaluationFunction == self.evaluate_CTS:
            currentScore = evaluationFunction(currentNodeSummary)
        else:
            currentScore = evaluationFunction(currentNodeSummary, control_name=self.control_name)

        # Prune Stats
        maxAbsDiff = 0
        maxDiff = -1.
        bestTreatment = self.control_name
        suboptTreatment = self.control_name
        maxDiffTreatment = self.control_name
        maxDiffSign = 0
        for treatment_group in currentNodeSummary:
            if treatment_group != self.control_name:
                diff = (currentNodeSummary[treatment_group][0]
                        - currentNodeSummary[self.control_name][0])
                if abs(diff) >= maxAbsDiff:
                    maxDiffTreatment = treatment_group
                    maxDiffSign = np.sign(diff)
                    maxAbsDiff = abs(diff)
                if diff >= maxDiff:
                    maxDiff = diff
                    suboptTreatment = treatment_group
                    if diff > 0:
                        bestTreatment = treatment_group
        if maxDiff > 0:
            pt = currentNodeSummary[bestTreatment][0]
            nt = currentNodeSummary[bestTreatment][1]
            pc = currentNodeSummary[self.control_name][0]
            nc = currentNodeSummary[self.control_name][1]
            p_value = (1. - stats.norm.cdf((pt - pc) / np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))) * 2
        else:
            pt = currentNodeSummary[suboptTreatment][0]
            nt = currentNodeSummary[suboptTreatment][1]
            pc = currentNodeSummary[self.control_name][0]
            nc = currentNodeSummary[self.control_name][1]
            p_value = (1. - stats.norm.cdf((pc - pt) / np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))) * 2
        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestAttribute = None

        # last column is the result/target column, 2nd to the last is the treatment group
        columnCount = X.shape[1]
        if (self.max_features and self.max_features > 0 and self.max_features <= columnCount):
            max_features = self.max_features
        else:
            max_features = columnCount

        for col in list(np.random.choice(a=range(columnCount), size=max_features, replace=False)):
            columnValues = X[:, col]
            # unique values
            lsUnique = np.unique(columnValues)

            if (isinstance(lsUnique[0], int) or
                isinstance(lsUnique[0], float)):
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = np.unique(lspercentile)

            for value in lsUnique:
                X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment, y, col, value)
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
                if set(leftNodeSummary.keys()) != set(rightNodeSummary.keys()):
                    continue
                node_mst = 10**8
                for ti in leftNodeSummary:
                    node_mst = np.min([node_mst, leftNodeSummary[ti][1]])
                    node_mst = np.min([node_mst, rightNodeSummary[ti][1]])
                if node_mst < min_samples_treatment:
                    continue
                # evaluate the split

                if evaluationFunction == self.evaluate_CTS:
                    leftScore1 = evaluationFunction(leftNodeSummary)
                    rightScore2 = evaluationFunction(rightNodeSummary)
                    gain = (currentScore - p * leftScore1 - (1 - p) * rightScore2)
                    gain_for_imp = (len(X) * currentScore - len(X_l) * leftScore1 - len(X_r) * rightScore2)
                else:
                    if (self.control_name in leftNodeSummary and
                        self.control_name in rightNodeSummary):
                        leftScore1 = evaluationFunction(leftNodeSummary, control_name=self.control_name)
                        rightScore2 = evaluationFunction(rightNodeSummary, control_name=self.control_name)
                        gain = (p * leftScore1 + (1 - p) * rightScore2 - currentScore)
                        gain_for_imp = (len(X_l) * leftScore1 + len(X_r) * rightScore2 - len(X) * currentScore)
                        if self.normalization:
                            norm_factor = self.normI(currentNodeSummary,
                                                     leftNodeSummary,
                                                     rightNodeSummary,
                                                     self.control_name,
                                                     alpha=0.9)
                        else:
                            norm_factor = 1
                        gain = gain / norm_factor
                    else:
                        gain = 0
                if (gain > bestGain and len(X_l) > min_samples_leaf and len(X_r) > min_samples_leaf):
                    bestGain = gain
                    bestAttribute = (col, value)
                    best_set_left = [X_l, w_l, y_l]
                    best_set_right = [X_r, w_r, y_r]
                    self.feature_imp_dict[bestAttribute[0]] += gain_for_imp

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(X)}
        # Add treatment size
        dcY['group_size'] = ''
        for treatment_group in currentNodeSummary:
            dcY['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        dcY['upliftScore'] = [round(upliftScore[0], 4), round(upliftScore[1], 4)]
        dcY['matchScore'] = round(upliftScore[0], 4)

        if bestGain > 0 and depth < max_depth:
            trueBranch = self.growDecisionTreeFrom(
                *best_set_left, evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )
            falseBranch = self.growDecisionTreeFrom(
                *best_set_right, evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )

            return DecisionTree(
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(treatment, y),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:
            if evaluationFunction == self.evaluate_CTS:
                return DecisionTree(
                    results=self.uplift_classification_results(treatment, y),
                    summary=dcY, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )
            else:
                return DecisionTree(
                    results=self.uplift_classification_results(treatment, y),
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
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    random_state: int, optional (default=2019)
        The seed used by the random number generator.

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
    
    n_jobs: int, optional (default=-1)
        The parallelization parameter to define how many parallel jobs need to be created. 
        This is passed on to joblib library for parallelizing uplift-tree creation.

    Outputs
    ----------
    df_res: pandas dataframe
        A user-level results dataframe containing the estimated individual treatment effect.
    """
    def __init__(self,
                 n_estimators=10,
                 max_features=10,
                 random_state=2019,
                 max_depth=5,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 evaluationFunction=None,
                 control_name=None,
                 normalization=True,
                 n_jobs=-1):
        """
        Initialize the UpliftRandomForestClassifier class.
        """
        self.classes_ = {}
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.evaluationFunction = evaluationFunction
        self.control_name = control_name
        self.n_jobs = n_jobs

        # Create forest
        self.uplift_forest = []
        for _ in range(n_estimators):
            uplift_tree = UpliftTreeClassifier(
                max_features=self.max_features, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg,
                evaluationFunction=self.evaluationFunction,
                control_name=self.control_name,
                normalization=normalization)

            self.uplift_forest.append(uplift_tree)

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
        np.random.seed(self.random_state)

        # Get treatment group keys
        treatment_group_keys = list(set(treatment))
        treatment_group_keys.remove(self.control_name)
        treatment_group_keys.sort()
        self.classes_ = {}
        for i, treatment_group_key in enumerate(treatment_group_keys):
            self.classes_[treatment_group_key] = i

        self.uplift_forest = (
            Parallel(n_jobs=self.n_jobs)
            (delayed(self.bootstrap)(X, treatment, y, tree) for tree in self.uplift_forest)
        )

        all_importances = [tree.feature_importances_ for tree in self.uplift_forest]
        self.feature_importances_ = np.mean(all_importances, axis=0)
        self.feature_importances_ /= self.feature_importances_.sum()  # normalize to add to 1

    @staticmethod
    def bootstrap(X, treatment, y, tree):
        bt_index = np.random.choice(len(X), len(X))
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
            An ndarray  containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.
        
        df_res : DataFrame, shape = [num_samples, (num_treatments + 1)]
            If full_output, a DataFrame containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.

        '''
        df_res = pd.DataFrame()
        y_pred_ensemble = dict()
        y_pred_list = np.zeros((X.shape[0], len(self.classes_)))

        # Make prediction by each tree
        for tree_i in range(len(self.uplift_forest)):

            _, _, _, y_pred_full = self.uplift_forest[tree_i].predict(X=X, full_output=True)

            if tree_i == 0:
                for treatment_group in y_pred_full:
                    y_pred_ensemble[treatment_group] = (
                        np.array(y_pred_full[treatment_group]) / len(self.uplift_forest)
                    )
            else:
                for treatment_group in y_pred_full:
                    y_pred_ensemble[treatment_group] = (
                        np.array(y_pred_ensemble[treatment_group])
                        + np.array(y_pred_full[treatment_group]) / len(self.uplift_forest)
                    )

        # Summarize results into dataframe
        for treatment_group in y_pred_ensemble:
            df_res[treatment_group] = y_pred_ensemble[treatment_group]

        df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)

        # Calculate delta
        delta_cols = []
        for treatment_group in y_pred_ensemble:
            if treatment_group != self.control_name:
                delta_cols.append('delta_%s' % (treatment_group))
                df_res['delta_%s' % (treatment_group)] = df_res[treatment_group] - df_res[self.control_name]
                # Add deltas to results list
                y_pred_list[:, self.classes_[treatment_group]] = df_res['delta_%s' % (treatment_group)].values
        df_res['max_delta'] = df_res[delta_cols].max(axis=1)

        if full_output:
            return df_res
        else:
            return y_pred_list
