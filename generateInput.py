import os
import argparse
import numpy as np
import pandas as pd

# python generateInput.py --data dataDirectory --metadata metadata --minPairs minimalNumberOfPairs --reference referenceEdges --output outputDirectory


parser = argparse.ArgumentParser ()
parser.add_argument ("--data", type = str, required = True, help = "Directory for TCGA miRNA and RNA counts")
parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing sample pairing and initial contexts")
parser.add_argument ("--minPairs", type = int, required = True, help = "Minimal number of tumor-normal sample pairs per context")
parser.add_argument ("--reference", type = str, required = True, help = "Table of reference edges, pointing from regulator to target (TSV)")
parser.add_argument ("--output", type = str, required = True, help = "Output directory for filtered raw value matrices, metadata and reference edge list")
args = parser.parse_args ()

if not os.path.exists (args.output):
    os.makedirs (args.output, exist_ok = True)

metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
numPairs = metadata.value_counts ("context"); allContexts = sorted (numPairs[numPairs > args.minPairs].index)
metadata = metadata.loc[metadata["context"].isin (allContexts)].reset_index (drop = True)
metadata.to_csv (os.path.join (args.output, "metadata.tsv"), index = None, sep = "\t")

mtx = pd.concat ([pd.read_csv (os.path.join (args.data, f"{context}_miRNA.tsv"), index_col = 0, sep = "\t") for context in allContexts], axis = 1)
pctExpressed = pd.DataFrame ({f"{C}_{T}": (mtx[metadata.loc[metadata["context"] == C, f"miRNA_{T}"]] > 5).mean (axis = 1) for C in allContexts for T in ["Tumor", "Normal"]})
allExpressed = list (pctExpressed.index[(pctExpressed > 0.5).all (axis = 1)]); miRNA_list = sorted (allExpressed)
print (f"overall expressed:\t{len (allExpressed)} miRNAs")

with np.errstate (divide = "ignore", invalid = "ignore"):
    numerator = np.log2 (pd.DataFrame (mtx.loc[miRNA_list, metadata["miRNA_Tumor"]].values, index = miRNA_list, columns = metadata["sample"].values))
    denominator = np.log2 (pd.DataFrame (mtx.loc[miRNA_list, metadata["miRNA_Normal"]].values, index = miRNA_list, columns = metadata["sample"].values))
log2FC = numerator - denominator
numerator.to_csv (os.path.join (args.output, "miRNA_log2tumor.tsv"), sep = "\t"); denominator.to_csv (os.path.join (args.output, "miRNA_log2normal.tsv"), sep = "\t")
log2FC.to_csv (os.path.join (args.output, "miRNA_log2FC.tsv"), sep = "\t")

mtx = pd.concat ([pd.read_csv (os.path.join (args.data, f"{context}_RNA.tsv"), index_col = 0, sep = "\t") for context in allContexts], axis = 1)
pctExpressed = pd.DataFrame ({f"{C}_{T}": (mtx[metadata.loc[metadata["context"] == C, f"RNA_{T}"]] > 5).mean (axis = 1) for C in allContexts for T in ["Tumor", "Normal"]})
allExpressed = list (pctExpressed.index[(pctExpressed > 0.5).all (axis = 1)]); RNA_list = sorted (allExpressed)
print (f"overall expressed:\t{len (allExpressed)} RNAs")

with np.errstate (divide = "ignore", invalid = "ignore"):
    numerator = np.log2 (pd.DataFrame (mtx.loc[RNA_list, metadata["RNA_Tumor"]].values, index = RNA_list, columns = metadata["sample"].values))
    denominator = np.log2 (pd.DataFrame (mtx.loc[RNA_list, metadata["RNA_Normal"]].values, index = RNA_list, columns = metadata["sample"].values))
log2FC = numerator - denominator
numerator.to_csv (os.path.join (args.output, "RNA_log2tumor.tsv"), sep = "\t"); denominator.to_csv (os.path.join (args.output, "RNA_log2normal.tsv"), sep = "\t")
log2FC.to_csv (os.path.join (args.output, "RNA_log2FC.tsv"), sep = "\t")

realEdges = pd.read_csv (args.reference, index_col = None, sep = "\t")
realEdges = pd.DataFrame ({"regulator": realEdges["mirnaName"].str.replace ("-3p", "").str.replace ("-5p", "").str.replace ("miR", "mir"),
                           "target": realEdges["geneName"]}).drop_duplicates (keep = "first")
realEdges = realEdges.loc[(realEdges["regulator"].isin (miRNA_list)) & (realEdges["target"].isin (RNA_list))].reset_index (drop = True)
realEdges.to_csv (os.path.join (args.output, "reference_edges.tsv"), index = None, sep = "\t")
print (realEdges.shape[0])


