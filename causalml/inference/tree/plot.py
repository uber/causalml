"""
Visualization functions for forest of trees-based ensemble methods for Uplift modeling on Classification
Problem.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pydotplus
import seaborn as sns

from .utils import get_tree_leaves_mask
from . import CausalTreeRegressor


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
    tree: CausalTreeRegressor, title: str = "Leaves values distribution"
) -> None:
    """
    Create distplot for tree leaves values
    Args:
        tree: (CausalTreeRegressor), Tree object
        figsize: (tuple), figure size
        title: (str), plot title

    Returns: None

    """
    tree_leaves_mask = get_tree_leaves_mask(tree)
    leaves_values = tree.tree_.value.reshape(
        -1,
    )[tree_leaves_mask]
    sns.distplot(leaves_values)
    plt.title(title)
    plt.show()
