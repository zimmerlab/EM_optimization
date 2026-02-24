PYTHON = python3.13
FUZZIFIER = /mnt/extproj/projekte/modelling/interactive_fuzzifier/cmdFuzzifier
DATA = /mnt/raidtmp/panc/expectation_maximization


.PHONY: generate_input fuzzify_regulator fuzzify_target EM_specific_only-suppressor EM_marker_only-suppressor EM_specific_sep-suppressor EM_marker_sep-suppressor
fuzzify: fuzzify_regulator fuzzify_target
EM_only-suppressor: EM_specific_only-suppressor EM_marker_only-suppressor
EM_sep-suppressor: EM_specific_sep EM_marker_sep-suppressor


generate_input: /mnt/raidtmp/panc/TCGA_counts/ /mnt/extproj/projekte/modelling/interactive_fuzzifier/example_input/metadata_paired.tsv /home/proj/projekte/annotation/mirna-binding-sites-klein/mirtarbase.isar.homo_sapiens.grch38.p14_ensembl-113.tsv
	$(PYTHON) generateInput.py --data /mnt/raidtmp/panc/TCGA_counts/ \
		--metadata /mnt/extproj/projekte/modelling/interactive_fuzzifier/example_input/metadata_paired.tsv \
		--minPairs 30 \
		--reference /home/proj/projekte/annotation/mirna-binding-sites-klein/mirtarbase.isar.homo_sapiens.grch38.p14_ensembl-113.tsv \
		--output /mnt/raidtmp/panc/expectation_maximization/raw_data/


fuzzify_regulator: $(DATA)/raw_data/miRNA_log2FC.tsv ./config/fuzzification.json
	$(PYTHON) $(FUZZIFIER)/estimator/main_estimator.py --mtx $(DATA)/raw_data/miRNA_log2FC.tsv \
		--config ./config/fuzzification.json \
		--output $(DATA)/concepts/miRNA_log2FC_concepts.json
	$(PYTHON) $(FUZZIFIER)/fuzzifier/main_fuzzifier.py --mtx $(DATA)/raw_data/miRNA_log2FC.tsv \
		--concept $(DATA)/concepts/miRNA_log2FC_concepts.json \
		--config ./config/fuzzification.json \
		--output $(DATA)/fuzzy_values/miRNA_log2FC/


fuzzify_target: $(DATA)/raw_data/RNA_log2FC.tsv ./config/fuzzification.json
	$(PYTHON) $(FUZZIFIER)/estimator/main_estimator.py --mtx $(DATA)/raw_data/RNA_log2FC.tsv \
		--config ./config/fuzzification.json \
		--output $(DATA)/concepts/RNA_log2FC_concepts.json
	$(PYTHON) $(FUZZIFIER)/fuzzifier/main_fuzzifier.py --mtx $(DATA)/raw_data/RNA_log2FC.tsv \
		--concept $(DATA)/concepts/RNA_log2FC_concepts.json \
		--config ./config/fuzzification.json \
		--output $(DATA)/fuzzy_values/RNA_log2FC/



EM_specific_only-suppressor: $(DATA)/fuzzy_values/ $(DATA)/raw_data/metadata.tsv ./output_only-suppressor/template/ ./output_only-suppressor/evaluation_template/ $(DATA)/raw_data/reference_edges.tsv ./config/EM_specific_only-suppressor.json
	$(PYTHON) main_EM.py --input $(DATA)/fuzzy_values/miRNA_log2FC/ $(DATA)/fuzzy_values/RNA_log2FC/ \
		--metadata $(DATA)/raw_data/metadata.tsv \
		--template ./output_only-suppressor/template/ \
		--evaluation_template ./output_only-suppressor/evaluation_template/ \
		--reference $(DATA)/raw_data/reference_edges.tsv \
		--config ./config/EM_specific_only-suppressor.json \
		--outputType context-specific \
		--output ./output_only-suppressor/context-specific/


EM_marker_only-suppressor: $(DATA)/fuzzy_values/ $(DATA)/raw_data/metadata.tsv ./output_only-suppressor/template/ ./output_only-suppressor/evaluation_template/ $(DATA)/raw_data/reference_edges.tsv ./config/EM_marker_only-suppressor.json
	$(PYTHON) main_EM.py --input $(DATA)/fuzzy_values/miRNA_log2FC/ $(DATA)/fuzzy_values/RNA_log2FC/ \
		--metadata $(DATA)/raw_data/metadata.tsv \
		--template ./output_only-suppressor/template/ \
		--evaluation_template ./output_only-suppressor/evaluation_template/ \
		--reference $(DATA)/raw_data/reference_edges.tsv \
		--config ./config/EM_marker_only-suppressor.json \
		--outputType marker \
		--output ./output_only-suppressor/marker/


EM_specific_sep-suppressor: $(DATA)/fuzzy_values/ $(DATA)/raw_data/metadata.tsv ./output_sep-suppressor/template/ $(DATA)/raw_data/reference_edges.tsv ./config/EM_specific_sep-suppressor.json
	$(PYTHON) main_EM.py --input $(DATA)/fuzzy_values/miRNA_log2FC/ $(DATA)/fuzzy_values/RNA_log2FC/ \
		--metadata $(DATA)/raw_data/metadata.tsv \
		--template ./output_sep-suppressor/template/ \
		--evaluation_template ./output_sep-suppressor/template/ \
		--reference $(DATA)/raw_data/reference_edges.tsv \
		--config ./config/EM_specific_sep-suppressor.json \
		--outputType context-specific \
		--output ./output_sep-suppressor/context-specific/


EM_marker_sep-suppressor: $(DATA)/fuzzy_values/ $(DATA)/raw_data/metadata.tsv ./output_sep-suppressor/template/ $(DATA)/raw_data/reference_edges.tsv ./config/EM_marker_sep-suppressor.json
	$(PYTHON) main_EM.py --input $(DATA)/fuzzy_values/miRNA_log2FC/ $(DATA)/fuzzy_values/RNA_log2FC/ \
		--metadata $(DATA)/raw_data/metadata.tsv \
		--template ./output_sep-suppressor/template/ \
		--evaluation_template ./output_sep-suppressor/template/ \
		--reference $(DATA)/raw_data/reference_edges.tsv \
		--config ./config/EM_marker_sep-suppressor.json \
		--outputType marker \
		--output ./output_sep-suppressor/marker/


