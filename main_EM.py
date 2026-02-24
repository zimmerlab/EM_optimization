import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from preparation import *
from expectation import expectation
from maximization import maximization
from evaluation import filterAssignments, plotScore, evaluation, plotSizes

# python main_EM.py --input fuzzyValueDirectory(s) --metadata metadata --template templateDirectory --evaluation_template evaluationTemplateDirectory \
#                   --reference referenceEdges --config config --outputType [context-specific / marker] --output outputDirectory


parser = argparse.ArgumentParser ()
parser.add_argument ("--input", nargs = "+", type = str, required = True, help = "Directory / directories for feature / regulator and target fuzzy values")
parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing initial contexts as well as regulator and target sample mapping (TSV)")
parser.add_argument ("--template", type = str, required = True, help = "Directory for template vectors / matrices (CSV)")
parser.add_argument ("--evaluation_template", type = str, required = True, help = "Directory for evaluation template vectors / matrices (CSV)")
parser.add_argument ("--reference", type = str, required = False, help = "Table of reference edges, pointing from regulator to target (TSV)")
parser.add_argument ("--config", type = str, required = True, help = "Config file for detailed parameters (JSON)")
parser.add_argument ("--outputType", type = str, required = True, choices = ["context-specific", "marker"], help = "Whether to deliver context-specific or marker optimization")
parser.add_argument ("--output", type = str, required = True, help = "Output directory for optimized feature / network and sample assignment")
args = parser.parse_args ()

metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
if metadata.columns[0] == "Unnamed: 0":
    metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
templateDict = readTemplates (args.template); allTemplates = sorted (templateDict.keys ())
evalTemplateDict = readTemplates (args.evaluation_template)
with open (args.config) as f:
    config = json.load (f)
mode = config["mode"]; allowExchange = config.get ("exchange_between_contexts", False)
colSep = config.get ("metadata_context_column_separator", "*")
cutoffDict = {"minimal_score_for_support": config["minimal_score_for_support"],
              "minimal_percent_for_support": config["minimal_percent_for_support"],
              "maximal_percent_for_not_support": config.get ("maximal_percent_for_not_support", np.inf)}
cutoffDict_context = {"minimal_score_for_support": 0.3,
                      "minimal_percent_for_support": config["minimal_percent_for_support"],
                      "maximal_percent_for_not_support": np.inf}
maxIter = config.get ("maximal_iterations", np.inf)

context = pd.DataFrame ({"sample": metadata[config.get ("metadata_index_column", "index")].values,
                         "context": metadata[config.get ("metadata_context_columns", ["context"])].agg (colSep.join, axis = 1).values,
                         "template": "unassigned"})
allContexts = sorted (set (context["context"])); sampleList = list (context["sample"])

if mode == "feature":
    scoreMtx, itemList = getFeatureScore (args.input[0], templateDict, sampleList, config)
elif mode == "edge":
    try:
        refEdges = pd.read_csv (args.reference, index_col = None, sep = "\t")
        refEdges = pd.DataFrame ({"regulator": refEdges[config.get ("reference_regulator_column", "regulator")].values,
                                  "target": refEdges[config.get ("reference_target_column", "target")].values})
    except FileNotFoundError:
        warnings.warn ("No reference edges available, using all combinatory edges instead.")
        refEdges = pd.DataFrame ()
    sampleMapping = pd.DataFrame ({"regulator": metadata[config.get ("metadata_regulator_column", "regulator")].values,
                                   "target": metadata[config.get ("metadata_target_column", "target")].values})
    scoreMtx, itemList, refEdges = getEdgeScore (args.input[0], args.input[1], refEdges, sampleMapping, templateDict, config)
else:
    raise ValueError ("Expectation-maximization algorithm only implemented for feautre-wise (\"feature\") or edge-wise (\"network\") optimization.")
print (f"scoring matrix completed: dimension {scoreMtx.shape}")

sampleLabeling = {temp: context.reset_index ().groupby ("context")["index"].agg (list).to_dict () for temp in allTemplates}
minContextSize = np.ceil (0.05 * context["context"].value_counts ())
history = context.rename (columns = {"context": "context_0"}); history["template_0"] = "unassigned"
blacklist = pd.DataFrame (False, index = context["sample"].values, columns = allContexts).reset_index (names = "sample")
markerAssignment = (args.outputType == "marker"); numIter = 1
print (f"expected output: {args.outputType} {mode}s")

if not os.path.exists (args.output):
    os.makedirs (args.output, exist_ok = True)
if not os.path.exists (os.path.join (args.output, "evaluation_plots")):
    os.makedirs (os.path.join (args.output, "evaluation_plots"), exist_ok = True)

while numIter < maxIter:
    print (f"----- iteration {numIter} -----")
    assignment, itemLabeling, allContexts = maximization (itemList, mode, sampleLabeling, scoreMtx, cutoffDict, allContexts, allTemplates,
                                                          uniqueTemplateAssignment = (numIter != 1), markerAssignment = markerAssignment)
    plotScore (scoreMtx, itemLabeling, sampleLabeling, allContexts, allTemplates, mode,
               os.path.join (args.output, "evaluation_plots", f"score_iteration_{numIter}.png"))
    context, sampleLabeling, allContexts, blacklist, addedBlacklist = expectation (sampleList, itemLabeling, scoreMtx, cutoffDict_context,
                                                                                   allContexts, allTemplates, history, blacklist,
                                                                                   allowExchange = allowExchange,
                                                                                   checkBlackList = (numIter > 5))
    numBlacklist = np.array (np.where (blacklist.drop ("sample", axis = 1))).shape[1]
#    if numIter > 2:
#        assignment, context = filterAssignments (assignment, context, allContexts, allTemplates, minContextSize)
#        sampleLabeling = {template: context.loc[context["template"] == template].groupby ("context")["index"].agg (list).to_dict ()
#                          for template in allTemplates}
#        allContexts = sorted (set (context["context"]))
    history, pctChanged, sameAssignment = evaluation (context, history, numIter)
    print ("edge assignment:"); print (assignment.value_counts ("context").sort_index ().to_dict ())
    print (f"sample assignment ({len (sampleList) - context.shape[0]} unassigned, {numBlacklist} blocked assignments):")
    print (context.value_counts ("context").sort_index ().to_dict ())
    print (f"percent changed: {pctChanged:.2%}")
    if sameAssignment and not addedBlacklist:
        break
    numIter += 1

if assignment.empty or context.empty:
    edgeScores = pd.concat ([pd.DataFrame (columns = [mode, "context", "template"], dtype = str),
                             pd.DataFrame (columns = sorted (evalTemplateDict.keys ()), dtype = float)],
                            axis = 1)
else:
    plotSizes (assignment, context, allContexts, allTemplates, mode, {"DOWN-UP": "tab:blue", "UP-DOWN": "tab:red"}, args.output)
    if mode == "feature":
        scoreMtx, itemList = getFeatureScore (args.input[0], evalTemplateDict, context["sample"].tolist (), config)
    elif mode == "edge":
        refEdges = assignment["edge"].str.split (config.get ("reference_edge_separator", "*"), expand = True)
        refEdges = refEdges.rename (columns = {0: "regulator", 1: "target"}); refEdges = refEdges.drop_duplicates ()
        sampleMapping = pd.DataFrame ({"regulator": context["sample"].values, "target": context["sample"].values})
        scoreMtx, itemList, _ = getEdgeScore (args.input[0], args.input[1], refEdges, sampleMapping, evalTemplateDict, config)
    else:
        raise ValueError ("Expectation-maximization algorithm only implemented for feautre-wise (\"feature\") or edge-wise (\"network\") optimization.")
    edgeScores = list (); allTemplates = sorted (evalTemplateDict.keys ())
    for cont in allContexts:
        tmp = pd.DataFrame (index = itemList, columns = allTemplates, dtype = float)
        for idx in range (len (allTemplates)):
            temp = allTemplates[idx]; idxList = context.loc[(context["context"] == cont) & (context["template"] == temp)].index
            if len (idxList) == 0:
                tmp[temp] = 0
            else:
                tmp[temp] = scoreMtx[idx, :, idxList].mean (axis = 0).round (3)
        tmp.insert (0, "context", cont); edgeScores.append (tmp.reset_index (names = mode))
    edgeScores = pd.concat (edgeScores, axis = 0, ignore_index = True)

assignment.to_csv (os.path.join (args.output, f"{mode}_assignment.tsv"), index = None, sep = "\t")
context.to_csv (os.path.join (args.output, "optimized_context_assignment.tsv"), index = None, sep = "\t")
history.set_index ("sample").rename_axis (None, axis = 0).to_csv (os.path.join (args.output, "context_assignment_history.tsv"), sep = "\t")
blacklist.to_csv (os.path.join (args.output, "context_assignment_blacklist.tsv"), index = None, sep = "\t")
edgeScores.to_csv (os.path.join (args.output, "evaluation_edge_score.tsv"), index = None, sep = "\t")


