import numpy as np
import pandas as pd
from assignment import assign


def maximization (itemList, itemLabel, labeling, scoreMtx, cutoffDict, allContexts, allTemplates, uniqueTemplateAssignment = False, markerAssignment = False):
    support_minScore = cutoffDict["minimal_score_for_support"]; assign_minPctSupport = cutoffDict["minimal_percent_for_support"]
    unassign_maxPctSupport = cutoffDict.get ("maximal_percent_for_not_support", np.inf)
    pre_assignment = assign (itemList, itemLabel, labeling, scoreMtx, support_minScore, assign_minPctSupport, unassign_maxPctSupport,
                             allContexts, allTemplates, uniqueTemplateAssignment = uniqueTemplateAssignment)
    if markerAssignment:
        df = pre_assignment.copy (); df["score"] = [pre_assignment.loc[idx, f"avgScore_{pre_assignment.loc[idx, 'template']}"] for idx in pre_assignment.index]
        occurrence = df.groupby (["index", itemLabel])["template"].value_counts ().reset_index (level = 2)
        df = df.set_index (["index", itemLabel]).sort_index (); tmp_list = list ()
        for idx in set (occurrence.index):
            tmpDF = df.loc[idx].sort_values ("score"); tmpOcc = occurrence.loc[idx]
            if tmpDF.shape[0] == 1:
                tmp_list.append (tmpDF.reset_index ().drop ("score", axis = 1))
            else:
                for temp in tmpOcc.loc[tmpOcc["count"] == 1, "template"]:
                    if temp == tmpDF["template"].iloc[-1]:
                        tmp_list.append (tmpDF.loc[tmpDF["template"] == temp].reset_index ().drop ("score", axis = 1))
        if len (tmp_list) == 0:
            pre_assignment = pd.concat ([pd.DataFrame ({"index": pd.Series (dtype = int)}),
                                         pd.DataFrame (columns = [itemLabel, "context", "template"], dtype = str),
                                         pd.DataFrame (columns = [f"avgScore_{temp}" for temp in allTemplates], dtype = float),
                                         pd.DataFrame (columns = [f"pctSupport_{temp}" for temp in allTemplates], dtype = float)],
                                        axis = 1)
        else:
            pre_assignment = pd.concat (tmp_list, axis = 0).sort_values (["index", "context", "template"]).reset_index (drop = True)
    if not pre_assignment.empty:
        tmp = pre_assignment.drop_duplicates (["index", itemLabel])
        with np.errstate (divide = "ignore", invalid = "ignore"):
            avgScore = np.array ([[(scoreMtx[i, :, labeling[allTemplates[i]].get (allContexts[j], list ())]).mean (axis = 0)
                                   for i in range (len (allTemplates))] for j in range (len (allContexts))])
        avgScore = pd.DataFrame (avgScore[:, :, tmp["index"]].max (axis = 1).T, index = tmp[itemLabel].values, columns = allContexts)
        assigned = pre_assignment.pivot (index = itemLabel, columns = "context", values = "template").replace (np.nan, "")
        assigned = (assigned != "").rename_axis (None).rename_axis (None, axis = 1)
        if assigned.shape[1] < len (allContexts):
            assigned = pd.concat ([assigned, pd.DataFrame (False, index = assigned.index, columns = list (set (allContexts) - set (assigned.columns)))], axis = 1)
        assigned = assigned.loc[avgScore.index, avgScore.columns]
        bestScore = pd.DataFrame ({"unassigned": avgScore.mask (assigned).max (axis = 1, skipna = True),
                                   "assigned": avgScore.mask (~assigned).min (axis = 1, skipna = True)})
        items = bestScore.loc[(bestScore["assigned"] > support_minScore) & (bestScore["unassigned"] < support_minScore)].index
        pre_assignment = pre_assignment.loc[pre_assignment[itemLabel].isin (items)]
        tmp = avgScore.loc[items]
    assignment = list ()
    for temp in allTemplates:
        sub_assignment = pre_assignment.loc[pre_assignment["template"] == temp]
        maxSize = pd.Series ([25 * len (labeling[temp].get (context, list ())) for context in allContexts], index = allContexts)
        assignment += [sub_assignment.loc[sub_assignment["context"] == C].sort_values ([f"pctSupport_{temp}", f"avgScore_{temp}"], ascending = False).head (maxSize[C])
                       for C in allContexts]
    if len (assignment) == 0:
        assignment = pd.concat ([pd.DataFrame ({"index": pd.Series (dtype = int)}),
                                 pd.DataFrame (columns = [itemLabel, "context", "template"], dtype = str),
                                 pd.DataFrame (columns = [f"avgScore_{temp}" for temp in allTemplates], dtype = float),
                                 pd.DataFrame (columns = [f"pctSupport_{temp}" for temp in allTemplates], dtype = float)],
                                axis = 1)
    else:
        assignment = pd.concat (assignment, axis = 0).sort_values ("index").reset_index (drop = True)
    newLabeling = {template: assignment.loc[assignment["template"] == template].groupby ("context")["index"].agg (list).to_dict () for template in allTemplates}
    newContexts = sorted (set (assignment["context"]))
    return assignment, newLabeling, newContexts


