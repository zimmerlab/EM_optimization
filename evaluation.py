import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plotScore (scoreMtx, itemLabeling, sampleLabeling, allContexts, allTemplates, mode, outputPath):
    if len (allContexts) == 0:
        return
    with np.errstate (divide = "ignore", invalid = "ignore"):
        avgScore = np.array ([[(scoreMtx[i, :, sampleLabeling[allTemplates[i]].get (allContexts[j], list ())]).mean (axis = 0)
                               for i in range (len (allTemplates))] for j in range (len (allContexts))])
    fig, axs = plt.subplots (len (allContexts), len (allTemplates), sharex = False, sharey = False, layout = "constrained",
                             figsize = (3 * len (allTemplates), 2 * len (allContexts)))
    for i in range (len (allContexts)):
        cont = allContexts[i]
        for j in range (len (allTemplates)):
            temp = allTemplates[j]
            if cont in itemLabeling[temp]:
                num = len (itemLabeling[temp][cont])
                values = pd.DataFrame (avgScore[:, j, itemLabeling[temp][cont]].T, index = itemLabeling[temp][cont], columns = allContexts)
                values = values.reset_index ().melt (id_vars = "index", var_name = "context")
                sns.lineplot (values, x = "context", y = "value", hue = "index", palette = {idx: "tab:blue" for idx in values["index"]},
                              legend = None, ax = axs[i, j])
                axs[i, j].set_xticks (axs[i, j].get_xticks ()); axs[i, j].set_xticklabels (axs[i, j].get_xticklabels (), rotation = 60, ha = "right")
            else:
                axs[i, j].set_xticks (list ()); axs[i, j].set_xticklabels (list ()); num = 0
            axs[i, j].set_xlabel (""); axs[i, j].set_ylabel (""); axs[i, j].set_ylim ((0, 1.05))
            axs[i, j].set_title (f"{cont} - {num} {temp}", size = 12)
    fig.supylabel (f"{mode} score", size = 10)
    plt.savefig (outputPath); plt.close ()



def filterAssignments (assignment, context, allContexts, allTemplates, minContextSize):
    newAssignment = assignment.copy (); newContext = context.copy ()
    for cont in allContexts:
        for temp in allTemplates:
            itemList = newAssignment.loc[(newAssignment["context"] == cont) & (newAssignment["template"] == temp)].index
            sampleList = newContext.loc[(newContext["context"] == cont) & (newContext["template"] == temp)].index
            if len (itemList) < 5 or len (sampleList) < minContextSize[cont]:
                newAssignment.loc[itemList, "context"] = "unassigned"
                newContext.loc[sampleList, "context"] = "unassigned"
    newAssignment = newAssignment.loc[newAssignment["context"] != "unassigned"].reset_index (drop = True)
    newContext = newContext.loc[newContext["context"] != "unassigned"].reset_index (drop = True)
    return newAssignment, newContext



def evaluation (context, history, numIter):
    newHistory = pd.concat ([history, pd.DataFrame ("unassigned", index = history.index, columns = [f"context_{numIter}", f"template_{numIter}"])], axis = 1)
    newHistory.loc[context["index"], f"context_{numIter}"] = context["context"].values
    newHistory.loc[context["index"], f"template_{numIter}"] = context["template"].values
    pctChanged = (newHistory[f"context_{numIter - 1}"] != newHistory[f"context_{numIter}"]).mean ()
    sameAssignment = False
    for i in range (1, numIter):
        if (newHistory[f"context_{i}"] == newHistory[f"context_{numIter}"]).all () and (newHistory[f"template_{i}"] == newHistory[f"template_{numIter}"]).all ():
            sameAssignment = True; break
    return newHistory, pctChanged, sameAssignment



def plotSizes (assignment, context, allContexts, allTemplates, itemLabel, palette, outputDir):
    ref = pd.DataFrame (itertools.product (allContexts, allTemplates), columns = ["context", "template"])
    size = assignment.value_counts (["context", "template"]).reset_index ()
    size = size.merge (ref, on = ["context", "template"], how = "right")
    size.loc[np.isnan (size["count"]), "count"] = 0; size["count"] = size["count"].astype (int)
    fig, ax = plt.subplots (figsize = (6, 4))
    sns.barplot (size, x = "context", y = "count", order = allContexts, hue = "template", hue_order = allTemplates, palette = palette, ax = ax)
    ax.set_xlabel (""); ax.set_ylabel (f"number of {itemLabel}s", size = 10)
    ax.bar_label (ax.containers[0], fontsize = 8); ax.bar_label (ax.containers[1], fontsize = 8); ax.legend (title = None)
    fig.tight_layout (); plt.savefig (os.path.join (outputDir, f"{itemLabel}_assignment.png")); plt.close ()
    size = context.value_counts (["context", "template"]).reset_index ()
    size = size.merge (ref, on = ["context", "template"], how = "right")
    size.loc[np.isnan (size["count"]), "count"] = 0; size["count"] = size["count"].astype (int)
    fig, ax = plt.subplots (figsize = (6, 4))
    sns.barplot (size, x = "context", y = "count", order = allContexts, hue = "template", hue_order = allTemplates, palette = palette, ax = ax)
    ax.set_xlabel (""); ax.set_ylabel ("number of sample pairs", size = 10)
    ax.bar_label (ax.containers[0], fontsize = 8); ax.bar_label (ax.containers[1], fontsize = 8); ax.legend (title = None)
    fig.tight_layout (); plt.savefig (os.path.join (outputDir, "optimized_context_assignment.png")); plt.close ()


