"""
Visualization functions for forest of trees-based ensemble methods for Uplift modeling on Classification
Problem. 
"""

from __future__ import print_function
from collections import defaultdict
import numpy as np
import pandas as pd
import pydotplus

def uplift_tree_string(decisionTree, x_names):
    '''
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
    '''
    
    ### Column Heading
    dcHeadings = {}
    for i, szY in enumerate(x_names + ['treatment_group_key']):
        szCol = 'Column %d' % i
        dcHeadings[szCol] = str(szY)
    
    def toString(decisionTree, indent=''):
        if decisionTree.results != None:  # leaf node
            return str(decisionTree.results)
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %s?' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s?' % (szCol, decisionTree.value)
            trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def uplift_tree_plot(decisionTree, x_names):
    '''
    Convert the tree to dot gragh for plots.

    Args
    ----

    decisionTree : object
        object of DecisionTree class
    
    x_names : list
        List of feature names

    Returns
    -------
    Dot class representing the tree graph. 
    '''
    
    
    ### Column Heading
    dcHeadings = {}
    for i, szY in enumerate(x_names + ['treatment_group_key']):
        szCol = 'Column %d' % i
        dcHeadings[szCol] = str(szY)
 
    dcNodes = defaultdict(list)
    """Plots the obtained decision tree. """

    def toString(iSplit, decisionTree, bBranch, szParent="null", indent=''):
        if decisionTree.results != None:  # leaf node
            lsY = []
            for szX, n in decisionTree.results.items():
                lsY.append('%s:%.2f' % (szX, n))
            dcY = {"name": "%s" % ', '.join(lsY), "parent": szParent}
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append(['leaf', dcY['name'], szParent, bBranch, str(-round(float(decisionTree.summary['impurity']),3)),
                                    dcSummary['samples'], dcSummary['group_size'], dcSummary['upliftScore'],dcSummary['matchScore']])
            return dcY
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %s' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s' % (szCol, decisionTree.value)
            trueBranch = toString(iSplit + 1, decisionTree.trueBranch, True, decision, indent + '\t\t')
            falseBranch = toString(iSplit + 1, decisionTree.falseBranch, False, decision, indent + '\t\t')
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append([iSplit + 1, decision, szParent, bBranch, str(-round(float(decisionTree.summary['impurity']),3)),
                                    dcSummary['samples'],dcSummary['group_size'], dcSummary['upliftScore'],dcSummary['matchScore']])
            return

    toString(0, decisionTree, None)
    lsDot = ['digraph Tree {',
             'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
             'edge [fontname=helvetica] ;'
             ]
    i_node = 0
    dcParent = {}
    for nSplit in range(len(dcNodes.items())):
        lsY = dcNodes[nSplit]
        for lsX in lsY:
            iSplit, decision, szParent, bBranch, szImpurity, szSamples, szGroup, upliftScore, matchScore = lsX
            if type(iSplit) == int:
                szSplit = '%d-%s' % (iSplit, decision)
                dcParent[szSplit] = i_node
                lsDot.append('%d [label=<%s<br/> impurity %s<br/> total_sample %s <br/>group_sample %s <br/> uplift score: %s <br/> uplift p_value %s <br/> validation uplift score %s>, fillcolor="#e5813900"] ;' % (i_node,
                                                                                                          decision.replace(
                                                                                                              '>=',
                                                                                                              '&ge;').replace(
                                                                                                              '?', ''),
                                                                                                          szImpurity,
                                                                                                          szSamples,
                                                                                                          szGroup,
                                                                                                          str(upliftScore[0]), 
                                                                                                          str(upliftScore[1]),
                                                                                                          str(matchScore)))
            else:
                lsDot.append('%d [label=< impurity %s<br/> total_sample %s <br/>group_sample %s <br/> uplift score: %s <br/> uplift p_value %s <br/> validation uplift score %s <br/> mean %s>, fillcolor="#e5813900"] ;' % (i_node,
                                                                                                                szImpurity,
                                                                                                                szSamples,
                                                                                                                szGroup,
                                                                                                                str(upliftScore[0]),
                                                                                                                str(upliftScore[1]),
                                                                                                                str(matchScore),
                                                                                                                decision))

            if szParent != 'null':
                if bBranch:
                    szAngle = '45'
                    szHeadLabel = 'True'
                else:
                    szAngle = '-45'
                    szHeadLabel = 'False'
                szSplit = '%d-%s' % (nSplit, szParent)
                p_node = dcParent[szSplit]
                if nSplit == 1:
                    lsDot.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                                                                    i_node, szAngle,
                                                                                                    szHeadLabel))
                else:
                    lsDot.append('%d -> %d ;' % (p_node, i_node))
            i_node += 1
    lsDot.append('}')
    dot_data = '\n'.join(lsDot)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph







