"""
Visualization functions for forest of trees-based ensemble methods for Uplift modeling on Classification
Problem.
"""

from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pydotplus
import seaborn as sns
from sklearn.tree import _tree
from sklearn.tree._export import _MPLTreeExporter, _color_brew
from sklearn.utils.validation import check_is_fitted

from . import CausalTreeRegressor
from .utils import get_tree_leaves_mask


def uplift_tree_string(decisionTree, x_names):
    """
    Convert the tree to string for print.

    Args
    ----

    decisionTree : object
        object of DecisionTree class

    x_names : list
        List of feature names

    Returns
    -------
    A string representation of the tree.
    """

    # Column Heading
    dcHeadings = {}
    for i, szY in enumerate(x_names + ["treatment_group_key"]):
        szCol = "Column %d" % i
        dcHeadings[szCol] = str(szY)

    def toString(decisionTree, indent=""):
        if decisionTree.results is not None:  # leaf node
            return str(decisionTree.results)
        else:
            szCol = "Column %s" % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(
                decisionTree.value, float
            ):
                decision = "%s >= %s?" % (szCol, decisionTree.value)
            else:
                decision = "%s == %s?" % (szCol, decisionTree.value)
            trueBranch = (
                indent + "yes -> " + toString(decisionTree.trueBranch, indent + "\t\t")
            )
            falseBranch = (
                indent + "no  -> " + toString(decisionTree.falseBranch, indent + "\t\t")
            )
            return decision + "\n" + trueBranch + "\n" + falseBranch

    print(toString(decisionTree))


def uplift_tree_plot(decisionTree, x_names):
    """
    Convert the tree to dot graph for plots.

    Args
    ----

    decisionTree : object
        object of DecisionTree class

    x_names : list
        List of feature names

    Returns
    -------
    Dot class representing the tree graph.
    """

    # Column Heading
    dcHeadings = {}
    for i, szY in enumerate(x_names + ["treatment_group_key"]):
        szCol = "Column %d" % i
        dcHeadings[szCol] = str(szY)

    dcNodes = defaultdict(list)
    """Plots the obtained decision tree. """

    def toString(
        iSplit,
        decisionTree,
        bBranch,
        szParent="null",
        indent="",
        indexParent=0,
        upliftScores=list(),
    ):
        if decisionTree.results is not None:  # leaf node
            lsY = []
            for tr, p in zip(decisionTree.classes_, decisionTree.results):
                lsY.append(f"{tr}:{p:.2f}")
            dcY = {"name": ", ".join(lsY), "parent": szParent}
            dcSummary = decisionTree.summary
            upliftScores += [dcSummary["matchScore"]]
            dcNodes[iSplit].append(
                [
                    "leaf",
                    dcY["name"],
                    szParent,
                    bBranch,
                    str(-round(float(decisionTree.summary["impurity"]), 3)),
                    dcSummary["samples"],
                    dcSummary["group_size"],
                    dcSummary["upliftScore"],
                    dcSummary["matchScore"],
                    indexParent,
                ]
            )
        else:
            szCol = "Column %s" % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(
                decisionTree.value, float
            ):
                decision = "%s >= %s" % (szCol, decisionTree.value)
            else:
                decision = "%s == %s" % (szCol, decisionTree.value)

            indexOfLevel = len(dcNodes[iSplit])
            toString(
                iSplit + 1,
                decisionTree.trueBranch,
                True,
                decision,
                indent + "\t\t",
                indexOfLevel,
                upliftScores,
            )
            toString(
                iSplit + 1,
                decisionTree.falseBranch,
                False,
                decision,
                indent + "\t\t",
                indexOfLevel,
                upliftScores,
            )
            dcSummary = decisionTree.summary
            upliftScores += [dcSummary["matchScore"]]
            dcNodes[iSplit].append(
                [
                    iSplit + 1,
                    decision,
                    szParent,
                    bBranch,
                    str(-round(float(decisionTree.summary["impurity"]), 3)),
                    dcSummary["samples"],
                    dcSummary["group_size"],
                    dcSummary["upliftScore"],
                    dcSummary["matchScore"],
                    indexParent,
                ]
            )

    upliftScores = list()
    toString(0, decisionTree, None, upliftScores=upliftScores)

    upliftScoreToColor = dict()
    try:
        # calculate colors for nodes based on uplifts
        minUplift = min(upliftScores)
        maxUplift = max(upliftScores)
        upliftLevels = [
            (uplift - minUplift) / (maxUplift - minUplift) for uplift in upliftScores
        ]  # min max scaler
        baseUplift = float(decisionTree.summary.get("matchScore"))
        baseUpliftLevel = (baseUplift - minUplift) / (
            maxUplift - minUplift
        )  # min max scaler normalization
        white = np.array([255.0, 255.0, 255.0])
        blue = np.array([31.0, 119.0, 180.0])
        green = np.array([0.0, 128.0, 0.0])
        for i, upliftLevel in enumerate(upliftLevels):
            if upliftLevel >= baseUpliftLevel:  # go blue
                color = upliftLevel * blue + (1 - upliftLevel) * white
            else:  # go green
                color = (1 - upliftLevel) * green + upliftLevel * white
            color = [int(c) for c in color]
            upliftScoreToColor[upliftScores[i]] = ("#%2x%2x%2x" % tuple(color)).replace(
                " ", "0"
            )  # color code
    except Exception as e:
        print(e)

    lsDot = [
        "digraph Tree {",
        'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
        "edge [fontname=helvetica] ;",
    ]
    i_node = 0
    dcParent = {}
    totalSample = int(
        decisionTree.summary.get("samples")
    )  # initialize the value with the total sample size at root
    for nSplit in range(len(dcNodes.items())):
        lsY = dcNodes[nSplit]
        indexOfLevel = 0
        for lsX in lsY:
            (
                iSplit,
                decision,
                szParent,
                bBranch,
                szImpurity,
                szSamples,
                szGroup,
                upliftScore,
                matchScore,
                indexParent,
            ) = lsX

            sampleProportion = round(int(szSamples) * 100.0 / totalSample, 1)
            if type(iSplit) is int:
                szSplit = "%d-%d" % (iSplit, indexOfLevel)
                dcParent[szSplit] = i_node
                lsDot.append(
                    "%d [label=<%s<br/> impurity %s<br/> total_sample %s (%s&#37;)<br/>group_sample %s <br/> "
                    "uplift score: %s <br/> uplift p_value %s <br/> "
                    'validation uplift score %s>, fillcolor="%s"] ;'
                    % (
                        i_node,
                        decision.replace(">=", "&ge;").replace("?", ""),
                        szImpurity,
                        szSamples,
                        str(sampleProportion),
                        szGroup,
                        str(upliftScore[0]),
                        str(upliftScore[1]),
                        str(matchScore),
                        upliftScoreToColor.get(matchScore, "#e5813900"),
                    )
                )
            else:
                lsDot.append(
                    "%d [label=< impurity %s<br/> total_sample %s (%s&#37;)<br/>group_sample %s <br/> "
                    "uplift score: %s <br/> uplift p_value %s <br/> validation uplift score %s <br/> "
                    'mean %s>, fillcolor="%s"] ;'
                    % (
                        i_node,
                        szImpurity,
                        szSamples,
                        str(sampleProportion),
                        szGroup,
                        str(upliftScore[0]),
                        str(upliftScore[1]),
                        str(matchScore),
                        decision,
                        upliftScoreToColor.get(matchScore, "#e5813900"),
                    )
                )

            if szParent != "null":
                if bBranch:
                    szAngle = "45"
                    szHeadLabel = "True"
                else:
                    szAngle = "-45"
                    szHeadLabel = "False"
                szSplit = "%d-%d" % (nSplit, indexParent)
                p_node = dcParent[szSplit]
                if nSplit == 1:
                    lsDot.append(
                        '%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;'
                        % (p_node, i_node, szAngle, szHeadLabel)
                    )
                else:
                    lsDot.append("%d -> %d ;" % (p_node, i_node))
            i_node += 1
            indexOfLevel += 1
    lsDot.append("}")
    dot_data = "\n".join(lsDot)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph


def plot_dist_tree_leaves_values(
    tree: CausalTreeRegressor,
    title: str = "Leaves values distribution",
    figsize: tuple = (5, 5),
    fontsize: int = 12,
) -> None:
    """
    Create distplot for tree leaves values
    Args:
        tree: (CausalTreeRegressor), Tree object
        title: (str), plot title
        figsize: (tuple), figure size
        fontsize: (int), title font size

    Returns: None

    """
    tree_leaves_mask = get_tree_leaves_mask(tree)
    leaves_values = tree.tree_.value
    treatment_effects = leaves_values[:, 1] - leaves_values[:, 0]
    treatment_effects = treatment_effects.reshape(
        -1,
    )[tree_leaves_mask]
    fig, ax = plt.subplots(figsize=figsize)
    sns.distplot(
        treatment_effects,
        ax=ax,
    )
    plt.title(title, fontsize=fontsize)
    plt.show()


class _MPLCTreeExporter(_MPLTreeExporter):
    def __init__(
        self,
        causal_tree: CausalTreeRegressor,
        max_depth: int,
        feature_names: list,
        class_names: list,
        label: str,
        filled: bool,
        impurity: bool,
        groups_count: bool,
        treatment_groups: tuple,
        node_ids: bool,
        proportion: bool,
        rounded: bool,
        precision: int,
        fontsize: int,
    ):
        """
        Causal Tree exporter for matplotlib
        Source: https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/tree/_export.py
        Args:
            causal_tree: CausalTreeRegressor
                    The causal tree to be plotted
             max_depth:  int, default=None
                The maximum depth of the representation. If None, the tree is fully generated.
            feature_names: list of strings, default=None
                    Names of each of the features.
                    If None, generic names will be used ("X[0]", "X[1]", ...).
            class_names:    list of str or bool, default=None
                    Names of each of the target classes in ascending numerical order.
                    Only relevant for classification and not supported for multi-output.
                    If ``True``, shows a symbolic representation of the class name.
            label:  {'all', 'root', 'none'}, default='all'
                    Whether to show informative labels for impurity, etc.
                    Options include 'all' to show at every node, 'root' to show only at
                    the top root node, or 'none' to not show at any node.
            filled: bool, default=False
                    When set to ``True``, paint nodes to indicate extremity of node values
            impurity: bool, default=True
                    When set to ``True``, show the impurity at each node.
             groups_count: bool, default=True
                Add the number of treatment and control groups
            treatment_groups: tuple, default=(0, 1)
                    Treatment and control groups labels
            node_ids: bool, default=False
                    When set to ``True``, show the ID number on each node.
            proportion: bool, default=False
                When set to ``True``, change the display of 'values' and/or 'samples'
                to be proportions and percentages respectively.
            rounded: bool, default=False
                    When set to ``True``, draw node boxes with rounded corners and use
                    Helvetica fonts instead of Times-Roman.
            precision: int, default=3
                    Number of digits of precision for floating point in the values of
                    impurity, threshold and value attributes of each node.
            fontsize: int, default=None
                    Size of text font. If None, determined automatically to fit figure.
        """
        super().__init__(
            max_depth,
            feature_names,
            class_names,
            label,
            filled,
            impurity,
            node_ids,
            proportion,
            rounded,
            precision,
            fontsize,
        )
        self.causal_tree = causal_tree
        self.groups_count = groups_count
        self.treatment_groups = treatment_groups

    def node_to_str(
        self, tree: _tree.Tree, node_id: int, criterion: Union[str, object]
    ) -> str:
        """
        Generate the node content string
        Args:
            tree:      Tree class
            node_id:   int, Tree node id
            criterion: str or object, split criterion
        Returns: str, node content
        """
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            node_string += "%s %s %s%s" % (
                feature,
                characters[3],
                round(tree.threshold[node_id], self.precision),
                characters[4],
            )

        # Write impurity
        if self.impurity:
            if not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += "%s = " % criterion
            node_string += (
                str(round(tree.impurity[node_id], self.precision)) + characters[4]
            )

        # Write node sample count
        if labels:
            node_string += "samples = "
        if self.proportion:
            percent = (
                100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            )
            node_string += str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]

        # Write the number of samples per treatment and control groups
        if self.groups_count:
            for group in self.treatment_groups:
                node_string += (
                    f"Group {group} = {self.causal_tree._groups_cnt[node_id][group]} "
                )
        node_string += characters[4]

        # Write node class distribution / regression value
        if self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += "value = "
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype("S32")).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (
            self.class_names is not None
            and tree.n_classes[0] != 1
            and tree.n_outputs == 1
        ):
            # Only done for single-output classification trees
            if labels:
                node_string += "class = "
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(value),
                    characters[2],
                )
            node_string += class_name

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        return node_string + characters[5]

    def get_color(self, value: np.ndarray) -> str:
        """
        Compute HTML color for a Tree node
        Args:
            value: Tree node value
        Returns: str, html color code in #RRGGBB format
        """
        # Regression tree or multi-output
        color = list(self.colors["rgb"][0])
        alpha = float(value - self.colors["bounds"][0]) / (
            self.colors["bounds"][1] - self.colors["bounds"][0]
        )
        alpha = 0 if np.isnan(alpha) else alpha
        # Compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        return "#%2x%2x%2x" % tuple(color)

    def get_fill_color(self, tree: _tree.Tree, node_id: int) -> str:
        """
         Fetch appropriate color for node
        Args:
            tree:    Tree class
            node_id: int, node index
        Returns: str
        """
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (
                    np.nanmin(-tree.impurity),
                    np.nanmax(-tree.impurity),
                )
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.nanmin(tree.value), np.nanmax(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)


def plot_causal_tree(
    causal_tree: CausalTreeRegressor,
    *,
    max_depth: int = None,
    feature_names: list = None,
    class_names: list = None,
    label: str = "all",
    filled: bool = False,
    impurity: bool = True,
    groups_count: bool = True,
    treatment_groups: tuple = (0, 1),
    node_ids: bool = False,
    proportion: bool = False,
    rounded: bool = False,
    precision: int = 3,
    ax: plt.Axes = None,
    fontsize: int = None,
):
    """
    Plot a Causal Tree.
    Source: https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/tree/_export.py
    Args:
        causal_tree: CausalTreeRegressor
                The causal tree to be plotted
        max_depth:  int, default=None
                The maximum depth of the representation. If None, the tree is fully generated.
        feature_names: list of strings, default=None
                Names of each of the features.
                If None, generic names will be used ("X[0]", "X[1]", ...).
        class_names:    list of str or bool, default=None
                Names of each of the target classes in ascending numerical order.
                Only relevant for classification and not supported for multi-output.
                If ``True``, shows a symbolic representation of the class name.
        label:  {'all', 'root', 'none'}, default='all'
                Whether to show informative labels for impurity, etc.
                Options include 'all' to show at every node, 'root' to show only at
                the top root node, or 'none' to not show at any node.
        filled: bool, default=False
                When set to ``True``, paint nodes to indicate extremity of node values
        impurity: bool, default=True
                When set to ``True``, show the impurity at each node.
        groups_count: bool, default=True
                Add the number of treatment and control groups
        treatment_groups: tuple, default=(0, 1)
                Treatment and control groups labels
        node_ids: bool, default=False
                When set to ``True``, show the ID number on each node.
        proportion: bool, default=False
                When set to ``True``, change the display of 'values' and/or 'samples'
                to be proportions and percentages respectively.
        rounded: bool, default=False
                When set to ``True``, draw node boxes with rounded corners and use
                Helvetica fonts instead of Times-Roman.
        precision: int, default=3
                Number of digits of precision for floating point in the values of
                impurity, threshold and value attributes of each node.
        ax: matplotlib axis, default=None
                Axes to plot to. If None, use current axis. Any previous content
                is cleared.
        fontsize: int, default=None
                Size of text font. If None, determined automatically to fit figure.
    Returns:

    """
    check_is_fitted(causal_tree)

    exporter = _MPLCTreeExporter(
        causal_tree=causal_tree,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        groups_count=groups_count,
        treatment_groups=treatment_groups,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
    )
    exporter.export(causal_tree, ax=ax)
