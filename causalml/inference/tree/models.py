"""
Forest of trees-based ensemble methods for Uplift modeling on Classification
Problem. Those methods include random forests and extremely randomized trees.

The module structure is the following:
- The ``UpliftRandomForestClassifier`` base class implements different
  variants of uplift models based on random forest, with 'fit' and 'predict'
  method.
- The ``UpliftTreeClassifier`` base class implements the uplift trees (without
  Bootstraping for random forest), this class is called within
  ``UpliftRandomForestClassifier`` for constructing random forest.
"""

# Authors: Zhenyu Zhao <zhenyuz@uber.com>
#          Totte Harinen <totte@uber.com>

from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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
        The flase branch tree node (feature > value).

    results : dictionary
        The classification probability Pr(1) for each experiment group in the tree node.

    summary : dictionary
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.

    maxDiffTreatment : string
        The treatment name generating the maximum difference between treatment and control group.

    maxDiffSign : float
        The sign of the maxium difference (1. or -1.).

    nodeSummary : dictionary
        Summary statistics of the tree nodes {treatment: [y_mean, n]}, where y_mean stands for the target metric mean
        and n is the sample size.

    backupResults : dictionary
        The conversion proabilities in each treatment in the parent node {treatment: y_mean}. The parent node
        information is served as a backup for the children node, in case no valid statistics can be calculated from the
        children node, the parent node information will be used in certain cases.

    bestTreatment : string
        The treatment name providing the best uplift (treatment effect).

    upliftScore : list
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maxium treatment effect, and
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

    The uplift tree classifer is used in uplift random forest to construct the trees in the forest.

    Parameters
    ----------

    evaluationFunction : string
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    max_depth: int, optional (default=5)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=10)
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

        rows = [list(X[i]) + [treatment[i]] + [y[i]] for i in range(len(X))]
        resTree = self.growDecisionTreeFrom(
            rows, evaluationFunction=self.evaluationFunction,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            depth=1, min_samples_treatment=self.min_samples_treatment,
            n_reg=self.n_reg, parentNodeSummary=None
        )
        self.fitted_uplift_tree = resTree
        return self

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

        rows = [list(X[i]) + [treatment[i]] + [y[i]] for i in range(len(X))]
        self.pruneTree(rows,
                       tree=self.fitted_uplift_tree,
                       rule=rule,
                       minGain=minGain,
                       evaluationFunction=self.evaluationFunction,
                       notify=False,
                       n_reg=self.n_reg,
                       parentNodeSummary=None)
        return self

    def pruneTree(self, rows, tree, rule='maxAbsDiff', minGain=0.,
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

        minGain : float, optional (default = 0.0001)
            The minimum gain required to make a tree node split. The children tree branches are trimmed if the actual
            split gain is less than the minimum gain.

        rule : string, optional (default = 'maxAbsDiff')
            The prune rules. Supported values are 'maxAbsDiff' for optimizing the maximum absolute difference, and
            'bestUplift' for optimizing the node-size weighted treatment effect.

        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(
            rows, min_samples_treatment=self.min_samples_treatment,
            n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        (set1, set2) = self.divideSet(rows, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch.results is None:
            self.pruneTree(set1, tree.trueBranch, rule, minGain,
                           evaluationFunction, notify, n_reg,
                           parentNodeSummary=currentNodeSummary)
        if tree.falseBranch.results is None:
            self.pruneTree(set2, tree.falseBranch, rule, minGain,
                           evaluationFunction, notify, n_reg,
                           parentNodeSummary=currentNodeSummary)

        # merge leaves (potentionally)
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
                    set1, min_samples_treatment=self.min_samples_treatment,
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
                    set2, min_samples_treatment=self.min_samples_treatment,
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
                    set1, min_samples_treatment=self.min_samples_treatment,
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
                    set2, min_samples_treatment=self.min_samples_treatment,
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
                gain = ((1. * len(set1) / len(rows) * trueScoreD
                         + 1. * len(set2) / len(rows) * falseScoreD)
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
        assert len(X) == len(y) and len(X) != len(treatment), 'Data length must be equal for X, treatment, and y.'

        rows = [list(X[i]) + [treatment[i]] + [y[i]] for i in range(len(X))]
        self.fillTree(rows, tree=self.fitted_uplift_tree)
        return self

    def fillTree(self, rows, tree):
        """ Fill the data into an existing tree.

        This is a lower-level function to execute on the tree filling task.

        Args
        ----
        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        tree : object
            object of DecisionTree class

        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(rows, min_samples_treatment=0, n_reg=0, parentNodeSummary=None)
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        (set1, set2) = self.divideSet(rows, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch is not None:
            self.fillTree(set1, tree.trueBranch)
        if tree.falseBranch is not None:
            self.fillTree(set2, tree.falseBranch)

        # Update Information

        # matchScore
        matchScore = (currentNodeSummary[tree.bestTreatment][0] - currentNodeSummary[self.control_name][0])
        tree.matchScore = round(matchScore, 4)
        tree.summary['matchScore'] = round(matchScore, 4)

        # Samples, Group_size
        tree.summary['samples'] = len(rows)
        tree.summary['group_size'] = ''
        for treatment_group in currentNodeSummary:
            tree.summary['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(rows)
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

    def divideSet(self, rows, column, value):
        '''
        Tree node split.

        Args
        ----

        rows : list of list
               The internal data format.

        column : int
                The column used to split the data.

        value : float or int
                The value in the column for splitting the data.

        Returns
        -------
        (list1, list2) : list of list
                The left node (list of data) and the right node (list of data).
        '''
        splittingFunction = None

        # for int and float values
        if isinstance(value, int) or isinstance(value, float):
            splittingFunction = lambda row: row[column] >= value
        else:  # for strings
            splittingFunction = lambda row: row[column] == value
        list1 = [row for row in rows if splittingFunction(row)]
        list2 = [row for row in rows if not splittingFunction(row)]
        return (list1, list2)

    def group_uniqueCounts(self, rows):
        '''
        Count sample size by experiment group.

        Args
        ----

        rows : list of list
               The internal data format.

        Returns
        -------
        results : dictionary
                The control and treatment sample size.
        '''
        results = {}
        for row in rows:
            # treatment group in the 2nd last column
            r = row[-2]
            if r not in results:
                results[r] = {0: 0, 1: 0}
            results[r][row[-1]] += 1
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
        if qk < 0.1**6:
            qk = 0.1**6
        elif qk > 1-0.1**6:
            qk = 1-0.1**6
        S = pk * np.log(pk / qk) + (1-pk) * np.log((1-pk) / (1-qk))
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

    def tree_node_summary(self, rows, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        '''
        Tree node summary statistics.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

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
        results = self.group_uniqueCounts(rows)
        # node Summary: {treatment_group: [p(1), size]}
        nodeSummary = {}
        # iterate treatment group
        for r in results:
            n1 = results[r][1]
            ntot = results[r][0] + results[r][1]
            if parentNodeSummary is None:
                y_mean = 1.*n1/ntot
            elif ntot > min_samples_treatment:
                y_mean = 1. * (results[r][1] + parentNodeSummary[r][0] * n_reg) / (ntot + n_reg)
            else:
                y_mean = parentNodeSummary[r][0]
            nodeSummary[r] = [y_mean, ntot]
        return nodeSummary

    def uplift_classification_results(self, rows):
        '''
        Classification probability for each treatment in the tree node.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        Returns
        -------
        res : dictionary
            The probability of 1 in each treatment in the tree node.
        '''
        results = self.group_uniqueCounts(rows)
        res = {}
        for r in results:
            p = float(results[r][1]) / (results[r][0] + results[r][1])
            res[r] = round(p, 6)
        return res

    def growDecisionTreeFrom(self, rows, evaluationFunction, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None):
        '''
        Train the uplift decision tree.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

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

        if len(rows) == 0:
            return DecisionTree()

        # Current Node Info and Summary
        currentNodeSummary = self.tree_node_summary(
            rows, min_samples_treatment=min_samples_treatment, n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
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
        bestSets = None

        # last column is the result/target column, 2nd to the last is the treatment group
        columnCount = len(rows[0]) - 2
        if (self.max_features and self.max_features > 0 and self.max_features <= columnCount):
            max_features = self.max_features
        else:
            max_features = columnCount

        for col in list(np.random.choice(a=range(columnCount), size=max_features, replace=False)):
            columnValues = [row[col] for row in rows]
            # unique values
            lsUnique = list(set(columnValues))

            if (isinstance(lsUnique[0], int) or
                isinstance(lsUnique[0], float)):
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = list(set(lspercentile))

            for value in lsUnique:
                (set1, set2) = self.divideSet(rows, col, value)
                # check the split validity on min_samples_leaf  372
                if (len(set1) < min_samples_leaf or len(set2) < min_samples_leaf):
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                p = float(len(set1)) / len(rows)
                leftNodeSummary = self.tree_node_summary(
                    set1, min_samples_treatment=min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )

                rightNodeSummary = self.tree_node_summary(
                    set2, min_samples_treatment=min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=parentNodeSummary
                )
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
                else:
                    if (self.control_name in leftNodeSummary and
                        self.control_name in rightNodeSummary):
                        leftScore1 = evaluationFunction(leftNodeSummary, control_name=self.control_name)
                        rightScore2 = evaluationFunction(rightNodeSummary, control_name=self.control_name)
                        gain = (p * leftScore1 + (1 - p) * rightScore2 - currentScore)
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
                if (gain > bestGain and len(set1) > min_samples_leaf and
                    len(set2) > min_samples_leaf):
                    bestGain = gain
                    bestAttribute = (col, value)
                    bestSets = (set1, set2)

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(rows)}
        # Add treatment size
        dcY['group_size'] = ''
        for treatment_group in currentNodeSummary:
            dcY['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        dcY['upliftScore'] = [round(upliftScore[0], 4), round(upliftScore[1], 4)]
        dcY['matchScore'] = round(upliftScore[0], 4)

        if bestGain > 0 and depth < max_depth:
            trueBranch = self.growDecisionTreeFrom(
                bestSets[0], evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )
            falseBranch = self.growDecisionTreeFrom(
                bestSets[1], evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )

            return DecisionTree(
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(rows),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:
            if evaluationFunction == self.evaluate_CTS:
                return DecisionTree(
                    results=self.uplift_classification_results(rows),
                    summary=dcY, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )
            else:
                return DecisionTree(
                    results=self.uplift_classification_results(rows),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )

    def classify(self, observations, tree, dataMissing=False):
        '''
        Classifies (prediction) the observationss according to the tree.

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
            Classifies (prediction) the observationss according to the tree, assuming without missing data.

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
            Classifies (prediction) the observationss according to the tree, assuming with missing data.

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


def plot(decisionTree):
    '''
    Convert the tree to string for print.

    Args
    ----

    decisionTree : object
        object of DecisionTree class

    Returns
    -------
    A string representation of the tree.
    '''

    def toString(decisionTree, indent=''):
        '''
        Convert the tree to string for print.

        Args
        ----

        decisionTree : object
            object of DecisionTree class

        indent : string, optional (default = '')
            indent to separate the string.

        Returns
        -------
        A string representation of the tree.
        '''
        if decisionTree.results is not None:  # leaf node
            return str(decisionTree.results)
        else:
            szCol = 'Column %s' % decisionTree.col
            if (isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float)):
                decision = '%s >= %s?' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s?' % (szCol, decisionTree.value)
            trueBranch = (indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t'))
            falseBranch = (indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t'))
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def cat_group(dfx, kpix, n_group=10):
    '''
    Category Reduction for Categorical Variables

    Args
    ----

    dfx : dataframe
        The inputs data dataframe.

    kpix : string
        The column of the feature.

    n_group : int, optional (default = 10)
        The number of top category values to be remained, other category values will be put into "Other".

    Returns
    -------
    The transformed categorical feature value list.
    '''
    if dfx[kpix].nunique() > n_group:
        # get the top categories
        top = dfx[kpix].isin(dfx[kpix].value_counts().index[:n_group])
        dfx.loc[~top, kpix] = "Other"
        return dfx[kpix].values
    else:
        return dfx[kpix].values


def cat_transform(dfx, kpix, kpi1):
    '''
    Encoding string features.

    Args
    ----

    dfx : dataframe
        The inputs data dataframe.

    kpix : string
        The column of the feature.

    kpi1 : list
        The list of feature names.

    Returns
    -------
    dfx : DataFrame
        The updated dataframe containing the encoded data.

    kpi1 : list
        The updated feature names containing the new dummy feature names.
    '''
    df_dummy = pd.get_dummies(dfx[kpix].values)
    new_col_names = ['%s_%s' % (kpix, x) for x in df_dummy.columns]
    df_dummy.columns = new_col_names
    dfx = pd.concat([dfx, df_dummy], axis=1)
    for new_col in new_col_names:
        if new_col not in kpi1:
            kpi1.append(new_col)
    if kpix in kpi1:
        kpi1.remove(kpix)
    return dfx, kpi1


def cv_fold_index(n, i, k, random_seed=2018):
    '''
    Encoding string features.

    Args
    ----

    dfx : dataframe
        The inputs data dataframe.

    kpix : string
        The column of the feature.

    kpi1 : list
        The list of feature names.

    Returns
    -------
    dfx : DataFrame
        The updated dataframe containing the encoded data.

    kpi1 : list
        The updated feature names containing the new dummy feature names.
    '''
    np.random.seed(random_seed)
    rlist = np.random.choice(a=range(k), size=n, replace=True)
    fold_i_index = np.where(rlist == i)[0]
    return fold_i_index


# Categorize continuous variable
def cat_continuous(x, granularity='Medium'):
    '''
    Categorize (bin) continuous variable based on percentile.

    Args
    ----

    x : list
        Feature values.

    granularity : string, optional, (default = 'Medium')
        Control the granularity of the bins, optional values are: 'High', 'Medium', 'Low'.

    Returns
    -------
    res : list
        List of percentile bins for the feature value.
    '''
    if granularity == 'High':
        lspercentile = [np.percentile(x, 5),
                        np.percentile(x, 10),
                        np.percentile(x, 15),
                        np.percentile(x, 20),
                        np.percentile(x, 25),
                        np.percentile(x, 30),
                        np.percentile(x, 35),
                        np.percentile(x, 40),
                        np.percentile(x, 45),
                        np.percentile(x, 50),
                        np.percentile(x, 55),
                        np.percentile(x, 60),
                        np.percentile(x, 65),
                        np.percentile(x, 70),
                        np.percentile(x, 75),
                        np.percentile(x, 80),
                        np.percentile(x, 85),
                        np.percentile(x, 90),
                        np.percentile(x, 95),
                        np.percentile(x, 99)
                        ]
        res = ['> p90 (%s)' % (lspercentile[8]) if z > lspercentile[8] else
               '<= p10 (%s)' % (lspercentile[0]) if z <= lspercentile[0] else
               '<= p20 (%s)' % (lspercentile[1]) if z <= lspercentile[1] else
               '<= p30 (%s)' % (lspercentile[2]) if z <= lspercentile[2] else
               '<= p40 (%s)' % (lspercentile[3]) if z <= lspercentile[3] else
               '<= p50 (%s)' % (lspercentile[4]) if z <= lspercentile[4] else
               '<= p60 (%s)' % (lspercentile[5]) if z <= lspercentile[5] else
               '<= p70 (%s)' % (lspercentile[6]) if z <= lspercentile[6] else
               '<= p80 (%s)' % (lspercentile[7]) if z <= lspercentile[7] else
               '<= p90 (%s)' % (lspercentile[8]) if z <= lspercentile[8] else
               '> p90 (%s)' % (lspercentile[8]) for z in x]
    elif granularity == 'Medium':
        lspercentile = [np.percentile(x, 10),
                        np.percentile(x, 20),
                        np.percentile(x, 30),
                        np.percentile(x, 40),
                        np.percentile(x, 50),
                        np.percentile(x, 60),
                        np.percentile(x, 70),
                        np.percentile(x, 80),
                        np.percentile(x, 90)
                        ]
        res = ['<= p10 (%s)' % (lspercentile[0]) if z <= lspercentile[0] else
               '<= p20 (%s)' % (lspercentile[1]) if z <= lspercentile[1] else
               '<= p30 (%s)' % (lspercentile[2]) if z <= lspercentile[2] else
               '<= p40 (%s)' % (lspercentile[3]) if z <= lspercentile[3] else
               '<= p50 (%s)' % (lspercentile[4]) if z <= lspercentile[4] else
               '<= p60 (%s)' % (lspercentile[5]) if z <= lspercentile[5] else
               '<= p70 (%s)' % (lspercentile[6]) if z <= lspercentile[6] else
               '<= p80 (%s)' % (lspercentile[7]) if z <= lspercentile[7] else
               '<= p90 (%s)' % (lspercentile[8]) if z <= lspercentile[8] else
               '> p90 (%s)' % (lspercentile[8]) for z in x]
    else:
        lspercentile = [np.percentile(x, 15), np.percentile(x, 50), np.percentile(x, 85)]
        res = ['1-Very Low' if z < lspercentile[0] else
               '2-Low' if z < lspercentile[1] else
               '3-High' if z < lspercentile[2] else
               '4-Very High' for z in x]
    return res


def kpi_transform(dfx, kpi_combo, kpi_combo_new):
    '''
    Feature transformation from continuous feature to binned features for a list of features

    Args
    ----

    dfx : DataFrame
        DataFrame containing the features.

    kpi_combo : list of string
        List of feature names to be transformed

    kpi_combo_new : list of string
        List of new feature names to be assigned to the transformed features.

    Returns
    -------
    dfx : DataFrame
        Updated DataFrame containing the new features.
    '''
    for j in range(len(kpi_combo)):
        if type(dfx[kpi_combo[j]].values[0]) == str:
            dfx[kpi_combo_new[j]] = dfx[kpi_combo[j]].values
            dfx[kpi_combo_new[j]] = cat_group(dfx=dfx, kpix=kpi_combo_new[j])
        else:
            if len(kpi_combo) > 1:
                dfx[kpi_combo_new[j]] = cat_continuous(
                    dfx[kpi_combo[j]].values, granularity='Low'
                )
            else:
                dfx[kpi_combo_new[j]] = cat_continuous(
                    dfx[kpi_combo[j]].values, granularity='High'
                )
    return dfx


# Uplift Random Forests
class UpliftRandomForestClassifier:
    """ Uplift Random Forest for Classification Task.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the uplift random forest.

    evaluationFunction : string
        Choose from one of the models: 'TwoModel', 'XLearner', 'RLearner', 'KL', 'ED', 'Chi', 'CTS'.

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

    y_calibration_method: string, optional (default=None)
        Choose calibration method from 'isotonic', 'sigmoid' for calibrating
        the classification probability in meta-learners. Only applicable for
        'TwoModel', 'XLearner', 'RLearner' uplift models.

    w_calibration_method: string, optional (default='sigmoid')
        Choose calibration method from 'isotonic', 'sigmoid' for calibrating
        the treatment propensity in meta-learners. Only applicable for
        'XLearner', 'RLearner' uplift models.

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
                 y_calibration_method=None,
                 w_calibration_method='sigmoid'):
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
        self.y_calibration_method = y_calibration_method
        self.w_calibration_method = w_calibration_method
        if self.evaluationFunction == 'TwoModel':
            classification_tree = RandomForestClassifier(
                n_estimators=n_estimators, criterion='gini',
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, random_state=None
            )
            self.uplift_forest = [classification_tree]
        elif self.evaluationFunction == 'RLearner':
            classification_tree = RandomForestClassifier(
                n_estimators=n_estimators, criterion='gini',
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, random_state=None
            )
            regression_tree = RandomForestRegressor(
                n_estimators=n_estimators, criterion='mse',
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, random_state=None
            )
            self.mu_forest = [classification_tree]
            self.w_forest = [classification_tree]
            self.tau_forest = [regression_tree]
        elif self.evaluationFunction == 'XLearner':
            classification_tree = RandomForestClassifier(
                n_estimators=n_estimators, criterion='gini',
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, random_state=None
            )
            regression_tree = RandomForestRegressor(
                n_estimators=n_estimators, criterion='mse',
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                max_features=max_features, random_state=None
            )
            self.mu_forest = [classification_tree]
            self.tau_forest = [regression_tree]
        else:
            # Create Forest
            self.uplift_forest = []
            for tree_i in range(n_estimators):
                uplift_tree = UpliftTreeClassifier(
                    max_features=self.max_features, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_treatment=self.min_samples_treatment,
                    n_reg=self.n_reg,
                    evaluationFunction=self.evaluationFunction,
                    control_name=self.control_name,
                    normalization=normalization
                )
                self.uplift_forest.append(uplift_tree)

    def fit(self, X, treatment, y, nvcate=False, value=None, imp_cost=None, trigger_cost=None):
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

        nvcate : bool, optional (default=False)
            Whether or not the model uses net value optimization.

        value : list, optional (default=None)
            A list containing the value of conversion for each unit.

        imp_cost : dict, optional (default=None)
            A dict containing the impression cost of each treatment, ie a cost
            that is associated with each treatment event.

        trigger_cost : dict, optional (default=None)
            A dict containing the triggered cost of each treatment, ie a cost
            that is associated with a user taking an action (such as promo
            redemption) in the treatment group.
        """
        np.random.seed(self.random_state)
        # Get treatment group keys
        treatment_group_keys = list(set(treatment))
        treatment_group_keys.remove(self.control_name)
        treatment_group_keys.sort()
        self.classes_ = {}
        for i, treatment_group_key in enumerate(treatment_group_keys):
            self.classes_[treatment_group_key] = i

        # Two Model Approach
        if self.evaluationFunction == 'TwoModel':
            treatment_group_keys = list(set(treatment))
            model_RF = self.uplift_forest[0]
            self.uplift_forest = {}
            self.treatment_group_keys = treatment_group_keys
            for i in range(len(treatment_group_keys)):
                # Random Forests
                treatment_group_key = treatment_group_keys[i]
                # Clone random forests
                if self.y_calibration_method in ['isotonic', 'sigmoid']:
                    est = clone(model_RF, safe=True)
                    self.uplift_forest[treatment_group_key] = CalibratedClassifierCV(
                        est, cv=4, method=self.y_calibration_method
                    )
                else:
                    self.uplift_forest[treatment_group_key] = clone(model_RF, safe=True)
                data_index = [i for i, xi in enumerate(treatment) if xi == treatment_group_key]
                x_train = X[data_index]
                y_train = y[data_index]
                self.uplift_forest[treatment_group_key].fit(X=x_train, y=y_train)

        elif self.evaluationFunction in ['RLearner', 'XLearner']:
            # Get treatment and control group keys
            # Remove control group from the keys to prevent control-control
            # comparison
            treatment_group_keys = list(set(treatment))
            self.treatment_group_keys = treatment_group_keys
            treatment_group_keys.remove(self.control_name)
            # Turn experiment group labels into numbered categories
            w_cat = pd.Series(treatment, dtype='category')
            self.w_num = w_cat.cat.codes
            self.w_col = w_cat.cat.categories
            self.w_predictors = X
            # Load calibrated classifier for propensity score
            w_forest = RandomForestClassifier(n_estimators=100)
            w_forest_calib = CalibratedClassifierCV(
                w_forest, cv=4, method=self.w_calibration_method)
            self.w_forest_fit = w_forest_calib.fit(
                self.w_predictors, self.w_num)
            # R-Learner
            if self.evaluationFunction == 'RLearner':
                # Estimate propensity scores and get base regressors
                w_probs = pd.DataFrame(self.w_forest_fit.predict_proba(self.w_predictors))
                w_probs.columns = self.w_col
                mu_forest = self.mu_forest[0]
                tau_forest = self.tau_forest[0]
                self.tau_fit = {}
                # Net Value Optimization
                if value is None:
                    value = []
                if imp_cost is None:
                    imp_cost = {}
                if trigger_cost is None:
                    trigger_cost = {}
                if nvcate:
                    trigger_cost_l = np.array([trigger_cost[ti] for ti in treatment])
                    imp_cost_l = np.array([imp_cost[ti] for ti in treatment])

                # Iterate treatment groups to build models
                for i in range(len(treatment_group_keys)):
                    # Get X, W and Y data for the treatment vs control pair
                    treatment_group_key = treatment_group_keys[i]
                    treatment_index = [i for i, xi in enumerate(treatment) if xi == treatment_group_key]
                    control_index = [i for i, xi in enumerate(treatment) if xi == self.control_name]
                    data_index = control_index + treatment_index
                    x_train = X[data_index]
                    y_train = y[data_index]
                    w_train = np.zeros(len(X))
                    w_train[treatment_index] = 1
                    w_train = w_train[data_index]
                    w_hat = w_probs[treatment_group_key].iloc[data_index]
                    # Optional Y ~ X regressor calibration
                    if self.y_calibration_method in ['isotonic', 'sigmoid']:
                        est = clone(mu_forest, safe=True)
                        mu_reg = CalibratedClassifierCV(est, cv=4, method=self.y_calibration_method)
                    else:
                        mu_reg = clone(mu_forest, safe=True)
                    # Fit the optionally calibrated regressor and predict
                    # mu_hat
                    mu_reg_fit = mu_reg.fit(X=x_train, y=y_train)
                    mu_hat = mu_reg_fit.predict_proba(X=x_train)[:, 1]
                    # Optional net value optimization
                    if nvcate:
                        y_tilde = (
                            (value[data_index] - trigger_cost_l[data_index]) * y_train
                            - (value[data_index] - np.mean(trigger_cost_l[data_index])) * mu_hat
                            - (imp_cost_l[data_index] - np.mean(imp_cost_l[data_index]))
                        )
                    else:
                        y_tilde = y_train - mu_hat
                    # TODO: Establish a way to balance w_tilde
                    w_tilde = w_train - w_hat
                    if (w_tilde == 0).any():
                        raise ValueError("Some propensity scores are zero. Check that your sample size is "
                                         "sufficient for RandomForestClassifier with {} classes.".format(
                                             len(set(treatment)))
                                         )
                    pseudo_outcome = y_tilde / w_tilde
                    rlearner_weights = np.asarray(np.power(w_tilde, 2))
                    # Outcome regressor
                    tau_forest = clone(tau_forest, safe=True)
                    self.tau_fit[treatment_group_key] = tau_forest.fit(X=x_train, y=pseudo_outcome,
                                                                       sample_weight=rlearner_weights)

            # X-Learner
            elif self.evaluationFunction == 'XLearner':
                # X-Learner requires splitting the dataset into control and treatment
                x_train = X
                y_train = y
                X_0 = x_train[treatment == self.control_name]
                y_0 = y_train[treatment == self.control_name]
                mu_forest = self.mu_forest[0]
                tau_forest = self.tau_forest[0]
                self.tau_fit_0 = {}
                self.tau_fit_1 = {}
                self.tau_weight = {}
                # Fitting the model for each treatment against the control
                for i in range(len(treatment_group_keys)):
                    # Select treatment group observations
                    treatment_group_key = self.treatment_group_keys[i]
                    X_1 = x_train[treatment == treatment_group_key]
                    y_1 = y_train[treatment == treatment_group_key]
                    # Clone random forests
                    if self.y_calibration_method in ['isotonic', 'sigmoid']:
                        est = clone(mu_forest, safe=True)
                        mu_reg = CalibratedClassifierCV(
                            est, cv=4, method=self.y_calibration_method)
                    else:
                        mu_reg = clone(mu_forest, safe=True)
                    tau_reg = clone(tau_forest, safe=True)
                    # Estimate pseudo-residuals by crossing control and tretment trained models with control and
                    # treatment features
                    mu_hat_1 = mu_reg.fit(X=X_0, y=y_0).predict_proba(X_1)[:, 1]
                    mu_hat_0 = mu_reg.fit(X=X_1, y=y_1).predict_proba(X_0)[:, 1]
                    if nvcate:
                        value0 = value[treatment == self.control_name]
                        value1 = value[treatment == treatment_group_key]
                        pseudo_residual_1 = (
                            (value1 - trigger_cost[treatment_group_key]) * y_1
                            - (value1 - trigger_cost[self.control_name]) * mu_hat_1
                            - (imp_cost[treatment_group_key] - imp_cost[self.control_name])
                        )
                        pseudo_residual_0 = (
                            (value0 - trigger_cost[treatment_group_key]) * mu_hat_0
                            - (value0 - trigger_cost[self.control_name]) * y_0
                            - (imp_cost[treatment_group_key] - imp_cost[self.control_name])
                        )
                    else:
                        pseudo_residual_1 = y_1 - mu_hat_1
                        pseudo_residual_0 = mu_hat_0 - y_0

                    # Fit tau regressors and store models, together with
                    # weights g
                    self.tau_fit_0[treatment_group_key] = tau_reg.fit(X=X_0, y=pseudo_residual_0)
                    self.tau_fit_1[treatment_group_key] = tau_reg.fit(X=X_1, y=pseudo_residual_1)

        # Uplift Model Approach
        else:
            for tree_i in range(len(self.uplift_forest)):
                # Bootstrap
                bt_index = np.random.choice(len(X), len(X))
                x_train_bt = X[bt_index]
                y_train_bt = y[bt_index]
                treatment_train_bt = treatment[bt_index]
                self.uplift_forest[tree_i].fit(X=x_train_bt, treatment=treatment_train_bt, y=y_train_bt)
        return

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
        df_res : DataFrame, shape = [num_samples, (num_treatments + 1)]
            A DataFrame containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.

        '''
        df_res = pd.DataFrame()
        y_pred_ensemble = dict()
        y_pred_list = np.zeros((X.shape[0], len(self.classes_)))
        # Two Model Approach
        if self.evaluationFunction == 'TwoModel':
            x_test = X
            for i in range(len(self.treatment_group_keys)):
                treatment_group_key = self.treatment_group_keys[i]
                y_pred = self.uplift_forest[treatment_group_key].predict_proba(X=x_test)
                y_pred_ensemble[treatment_group_key] = [xi[1] for xi in y_pred]
        elif self.evaluationFunction == 'RLearner':
            x_test = X
            for i in range(len(self.treatment_group_keys)):
                treatment_group_key = self.treatment_group_keys[i]
                tau_fit = self.tau_fit[treatment_group_key]
                y_pred = tau_fit.predict(X=x_test)
                y_pred_ensemble[treatment_group_key] = [xi for xi in y_pred]
        elif self.evaluationFunction == 'XLearner':
            x_test = X
            control_name = self.control_name
            w_probs = pd.DataFrame(self.w_forest_fit.predict_proba(x_test))
            w_probs.columns = self.w_col
            control_prob = w_probs.loc[:, control_name]
            for i in range(len(self.treatment_group_keys)):
                treatment_group_key = self.treatment_group_keys[i]
                tau_fit_0 = self.tau_fit_0[treatment_group_key]
                tau_fit_1 = self.tau_fit_1[treatment_group_key]
                tau_0 = tau_fit_0.predict(X=x_test)
                tau_1 = tau_fit_1.predict(X=x_test)
                treatment_prob = w_probs.loc[:, treatment_group_key]
                y_pred = (
                    ((treatment_prob / (treatment_prob + control_prob)) * tau_0)
                    + ((control_prob / (treatment_prob + control_prob)) * tau_1)
                )
                y_pred_ensemble[treatment_group_key] = [xi for xi in y_pred]
        else:
            # Make prediction by each tree
            for tree_i in range(len(self.uplift_forest)):
                rec_treatment, y_pred_opt, upliftScores, y_pred_full = \
                    self.uplift_forest[tree_i].predict(X=X, full_output=True)
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

        # Summary results into dataframe
        for treatment_group in y_pred_ensemble:
            df_res[treatment_group] = y_pred_ensemble[treatment_group]
        if self.evaluationFunction in ['XLearner', 'RLearner']:
            neg_lift = df_res.max(axis=1) < 0
            df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)
            df_res.loc[neg_lift, 'recommended_treatment'] = self.control_name
        else:
            df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)

        # Calculating delta
        delta_cols = []
        if self.evaluationFunction in ['XLearner', 'RLearner']:
            for treatment_group in y_pred_ensemble:
                delta_cols.append('delta_%s' % (treatment_group))
                df_res['delta_%s' % (treatment_group)] = df_res[treatment_group]
                # add deltas to results list
                y_pred_list[:, self.classes_[treatment_group]] = df_res['delta_%s' % (treatment_group)].values
            df_res['max_delta'] = df_res[delta_cols].max(axis=1)
        else:
            for treatment_group in y_pred_ensemble:
                if treatment_group != self.control_name:
                    delta_cols.append('delta_%s' % (treatment_group))
                    df_res['delta_%s' % (treatment_group)] = df_res[treatment_group] - df_res[self.control_name]
                    # add deltas to results list
                    y_pred_list[:, self.classes_[treatment_group]] = df_res['delta_%s' % (treatment_group)].values
            df_res['max_delta'] = df_res[delta_cols].max(axis=1)

        return y_pred_list
