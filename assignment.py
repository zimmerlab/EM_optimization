import numpy as np
import pandas as pd


def assign (allItems, itemType, labeling, scoreMtx, support_minScore, assign_minPctSupport, unassign_maxPctSupport,
            allContexts, allTemplates, uniqueTemplateAssignment = True):
    assignment = list ()
    for context in allContexts:
        contextAssignment = list (); idxList = list ()
        for temp in allTemplates:
            if temp not in labeling:
                continue
            if context not in labeling[temp]:
                continue
            if itemType == "feature" or itemType == "edge":
                score = scoreMtx[:, :, labeling[temp][context]]
                pctSupport = pd.DataFrame ((score > support_minScore).mean (axis = 2).T, columns = allTemplates)
                supported = pctSupport.loc[(pctSupport[temp] > assign_minPctSupport) & (pctSupport.drop (temp, axis = 1) < unassign_maxPctSupport).all (axis = 1)]
                avgScore = pd.DataFrame (score[:, supported.index, :].mean (axis = 2).T, index = supported.index, columns = allTemplates)
            elif itemType == "sample":
                score = scoreMtx[:, labeling[temp][context], :]
                pctSupport = pd.DataFrame ((score > support_minScore).mean (axis = 1).T, columns = allTemplates)
                supported = pctSupport.loc[(pctSupport[temp] > assign_minPctSupport) & (pctSupport.drop (temp, axis = 1) < unassign_maxPctSupport).all (axis = 1)]
                avgScore = pd.DataFrame (score[:, :, supported.index].mean (axis = 1).T, index = supported.index, columns = allTemplates)
            else:
                raise ValueError
            infoMtx = pd.DataFrame ({"context": context, "template": temp}, index = avgScore.index)
            supported = pd.concat ([pd.DataFrame ({itemType: [allItems[x] for x in avgScore.index]}, index = avgScore.index), infoMtx,
                                    avgScore.rename (columns = {temp: f"avgScore_{temp}" for temp in allTemplates}).round (3),
                                    supported.rename (columns = {temp: f"pctSupport_{temp}" for temp in allTemplates}).round (3)],
                                   axis = 1)
            if not supported.empty:
                contextAssignment.append (supported.reset_index ())
        if len (contextAssignment) > 0:
            contextAssignment = pd.concat (contextAssignment, axis = 0, ignore_index = True)
            assigned = pd.DataFrame ({"template": contextAssignment["template"],
                                      "by_avgScore": contextAssignment.filter (regex = "^avgScore_").idxmax (axis = 1).str.split ("avgScore_", expand = True)[1],
                                      "by_pctSupport": contextAssignment.filter (regex = "^pctSupport_").idxmax (axis = 1).str.split ("pctSupport_", expand = True)[1]})
            contextAssignment = contextAssignment.loc[(assigned["template"] == assigned["by_avgScore"]) & (assigned["template"] == assigned["by_pctSupport"])]
            if uniqueTemplateAssignment:
                multiTemp = contextAssignment.value_counts ("index"); multiTemp = multiTemp[multiTemp > 1].index
                for idx in multiTemp:
                    tmp = contextAssignment.loc[contextAssignment["index"] == idx]
                    itemScore = pd.Series ([tmp.loc[i, f"avgScore_{tmp.loc[i, 'template']}"] for i in tmp.index], index = tmp.index).sort_values (ascending = False)
                    if itemScore.iloc[0] / itemScore.iloc[1] > 2:
                        idxList.append (itemScore.index[0])
                if len (idxList) == 0:
                    contextAssignment = contextAssignment.loc[~contextAssignment["index"].isin (multiTemp)]
                else:
                    contextAssignment = pd.concat ([contextAssignment.loc[~contextAssignment["index"].isin (multiTemp)], contextAssignment.loc[idxList]], axis = 0)
            if not contextAssignment.empty:
                assignment.append (contextAssignment)
    if len (assignment) == 0:
        assignment = pd.concat ([pd.DataFrame ({"index": pd.Series (dtype = int)}),
                                 pd.DataFrame (columns = [itemType, "context", "template"], dtype = str),
                                 pd.DataFrame (columns = [f"avgScore_{temp}" for temp in allTemplates], dtype = float),
                                 pd.DataFrame (columns = [f"pctSupport_{temp}" for temp in allTemplates], dtype = float)],
                                axis = 1)
    else:
        assignment = pd.concat (assignment, axis = 0).sort_values (["index", "context", "template"]).reset_index (drop = True)
    return assignment


