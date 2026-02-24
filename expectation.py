import numpy as np
import pandas as pd
from assignment import assign


def updateBlacklist (assignment, history, blacklist):
    numIter = int (history.columns[-1].replace ("template_", "")); newBlacklist = blacklist.copy (); addedBlacklist = False
    fullAssignment = assignment.merge (history[["sample"]], on = "sample", how = "right")[["context", "template"]].replace (np.nan, "unassigned")
    switching = (history.filter (regex = f"_{numIter - 4}$").values == history.filter (regex = f"_{numIter - 2}$").values).all (axis = 1)
    switching &= (history.filter (regex = f"_{numIter - 4}$").values != history.filter (regex = f"_{numIter - 3}$").values).any (axis = 1)
    switching &= (history.filter (regex = f"_{numIter - 3}$").values == history.filter (regex = f"_{numIter - 1}$").values).all (axis = 1)
    switching &= (history.filter (regex = f"_{numIter - 2}$").values == history.filter (regex = f"_{numIter}$").values).all (axis = 1)
    switching &= (history.filter (regex = f"_{numIter - 2}$").values != history.filter (regex = f"_{numIter - 1}$").values).any (axis = 1)
    switching &= (history.filter (regex = f"_{numIter - 1}$").values == fullAssignment.values).all (axis = 1)
    if switching.any ():
        addedBlacklist = True
        for idx in np.where (switching)[0]:
            assigned = assignment.loc[assignment["index"] == idx]
            if assigned.empty:
                context = history.loc[idx, f"context_{numIter}"]
            else:
                print (assigned.drop (["index", "template"], axis = 1))
                context = assigned["context"].iloc[0]; assignment = assignment.loc[assignment["index"] != idx].reset_index (drop = True)
            newBlacklist.loc[idx, context] = True
    return newBlacklist, addedBlacklist



def expectation (itemList, labeling, scoreMtx, cutoffDict, allContexts, allTemplates, history, blacklist, allowExchange = False, checkBlackList = False):
    support_minScore = cutoffDict["minimal_score_for_support"]; assign_minPctSupport = cutoffDict["minimal_percent_for_support"]
    unassign_maxPctSupport = cutoffDict.get ("maximal_percent_for_not_support", np.inf)
    pre_assignment = assign (itemList, "sample", labeling, scoreMtx, support_minScore, assign_minPctSupport, unassign_maxPctSupport,
                             allContexts, allTemplates, uniqueTemplateAssignment = True)
    restricted = blacklist.melt (id_vars = "sample", var_name = "context").merge (pre_assignment, on = ["sample", "context"], how = "right")
    pre_assignment = restricted.loc[~restricted["value"], pre_assignment.columns]
    if allowExchange:
        df = pre_assignment.copy (); df["score"] = [pre_assignment.loc[idx, f"avgScore_{pre_assignment.loc[idx, 'template']}"] for idx in pre_assignment.index]
        occurrence = df.groupby (["index", "sample"])["template"].value_counts ().reset_index (level = 2)
        df = df.set_index (["index", "sample"]).sort_index (); tmp_list = list ()
        for idx in set (occurrence.index):
            tmpDF = df.loc[idx].sort_values ("score"); tmpOcc = occurrence.loc[idx]
            if tmpDF.shape[0] == 1:
                tmp_list.append (tmpDF.reset_index ().drop ("score", axis = 1))
            else:
                for temp in tmpOcc.loc[tmpOcc["count"] == 1, "template"]:
                    if temp == tmpDF["template"].iloc[-1]:
                        tmp_list.append (tmpDF.loc[tmpDF["template"] == temp].reset_index ().drop ("score", axis = 1))
        if len (tmp_list) == 0:
            assignment = pd.concat ([pd.DataFrame ({"index": pd.Series (dtype = int)}),
                                     pd.DataFrame (columns = ["sample", "context", "template"], dtype = str),
                                     pd.DataFrame (columns = [f"avgScore_{temp}" for temp in allTemplates], dtype = float),
                                     pd.DataFrame (columns = [f"pctSupport_{temp}" for temp in allTemplates], dtype = float)],
                                    axis = 1)
        else:
            assignment = pd.concat (tmp_list, axis = 0).sort_values ("index").reset_index (drop = True)
    else:
        initContext = history[["sample", "context_0"]].rename (columns = {"context_0": "context"})
        assignment = pre_assignment.merge (initContext, on = ["sample", "context"], how = "inner").sort_values ("index").reset_index (drop = True)
        if assignment.empty:
            assignment = pd.concat ([pd.DataFrame ({"index": pd.Series (dtype = int)}),
                                     pd.DataFrame (columns = ["sample", "context", "template"], dtype = str),
                                     pd.DataFrame (columns = [f"avgScore_{temp}" for temp in allTemplates], dtype = float),
                                     pd.DataFrame (columns = [f"pctSupport_{temp}" for temp in allTemplates], dtype = float)],
                                    axis = 1)
    if checkBlackList:
        newBlacklist, addedBlacklist = updateBlacklist (assignment, history, blacklist)
    else:
        newBlacklist = blacklist.copy (); addedBlacklist = False
    newLabeling = {template: assignment.loc[assignment["template"] == template].groupby ("context")["index"].agg (list).to_dict () for template in allTemplates}
    newContexts = sorted (set (assignment["context"]))
    return assignment, newLabeling, newContexts, newBlacklist, addedBlacklist


