import os
import itertools
import numpy as np
import pandas as pd


def readFuzzyValues (dir, fileList, allSets, sampleList):
    allFV = list ()
    for file in fileList:
        memberships = pd.read_csv (os.path.join (dir, file), index_col = 0, sep = "\t")
        memberships = memberships.div (memberships.sum (axis = 1), axis = 0)
        allFV.append (memberships.loc[sampleList, allSets].to_numpy ())
    allFV = np.array (allFV)
    return allFV



def readTemplates (dir):
    templateDict = {file[:-4]: pd.read_csv (os.path.join (dir, file), header = None, index_col = None, sep = ",").to_numpy ()
                    for file in sorted (os.listdir (dir))}
    return templateDict



def getFeatureScore (fuzzyValueDir, templateDict, sampleList, config):
    allSets = config["fuzzy_sets"]; affix = config.get ("fuzzy_value_file_affix", {"prefix": "", "suffix": ""})
    featureList = [file.replace (affix["prefix"], "").replace (affix["suffix"], "").replace (".tsv", "") for file in os.listdir (fuzzyValueDir)]
    featureFV = readFuzzyValues (fuzzyValueDir, os.listdir (fuzzyValueDir), allSets, sampleList = sampleList)
    print (f"minimum: {featureFV.min (axis = None)}\tmaximum: {featureFV.max (axis = None)}")
    scoreMtx = list ()
    for temp in sorted (templateDict.keys ()):
        penalty = np.sqrt (((featureFV - templateDict[temp]) ** 2).sum (axis = 2))
        penalty[(featureFV == 0).all (axis = 2)] = np.nan
        scoreMtx.append (1 - penalty / np.sqrt (2))
    scoreMtx = np.array (scoreMtx)
    return scoreMtx, featureList



def getEdgeScore (regulatorDir, targetDir, refEdges, sampleMapping, templateDict, config):
    sep = config.get ("reference_edge_separator", "*")
    regulatorSets = config["regulator_fuzzy_sets"]; targetSets = config["target_fuzzy_sets"]
    regPrefix = config.get ("fuzzy_value_file_affix", dict ()).get ("regulator_prefix", "")
    regSuffix = config.get ("fuzzy_value_file_affix", dict ()).get ("regulator_suffix", "")
    tarPrefix = config.get ("fuzzy_value_file_affix", dict ()).get ("target_prefix", "")
    tarSuffix = config.get ("fuzzy_value_file_affix", dict ()).get ("target_suffix", "")
    if refEdges.empty:
        regulatorList = [file.replace (regPrefix, "").replace (regSuffix, "").replace (".tsv", "") for file in regulatorDir]
        targetList = [file.replace (tarPrefix, "").replace (tarSuffix, "").replace (".tsv", "") for file in targetDir]
        reference = pd.DataFrame (itertools.product (regulatorList, targetList), columns = ["regulator", "target"])
    else:
        reference = refEdges.copy ()
    reference["edge"] = reference["regulator"] + sep + reference["target"]; itemList = list (reference["edge"])
    regulatorList = sorted (set (reference["regulator"])); targetList = sorted (set (reference["target"]))
    regFV = readFuzzyValues (regulatorDir, [f"{regPrefix}{regulator}{regSuffix}.tsv" for regulator in regulatorList], regulatorSets,
                             sampleList = list (sampleMapping["regulator"]))
    tarFV = readFuzzyValues (targetDir, [f"{tarPrefix}{target}{tarSuffix}.tsv" for target in targetList], targetSets,
                             sampleList = list (sampleMapping["target"]))
    print (f"regulator\tminimum: {regFV.min (axis = None)}\tmaximum: {regFV.max (axis = None)}")
    print (f"target\t\tminimum: {tarFV.min (axis = None)}\tmaximum: {tarFV.max (axis = None)}")
    regIdx = pd.Series (range (len (regulatorList)), index = regulatorList); tmpRegFV = regFV[regIdx[reference["regulator"]], :, :]
    tarIdx = pd.Series (range (len (targetList)), index = targetList); tmpTarFV = tarFV[tarIdx[reference["target"]], :, :]
    scoreMtx = list ()
    for temp in sorted (templateDict.keys ()):
        penalty = np.sqrt (((np.einsum ("ijk, kl -> ijl", tmpRegFV, templateDict[temp]) - tmpTarFV) ** 2).sum (axis = 2))
        penalty[(tmpRegFV == 0).all (axis = 2) | (tmpTarFV == 0).all (axis = 2)] = np.sqrt (2)
        scoreMtx.append (1 - penalty / np.sqrt (2))
    scoreMtx = np.array (scoreMtx)
    return scoreMtx, itemList, reference


