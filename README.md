# The ensemble-binclass: an Python package for a feature selection and ensemble classification of molecular omic data
## Description
ensemble-binclass is an python package for single feature selection (FS) and ensemble classification of omic data (numeric data formats).
This tool is based on several feature filters, such as ... and classifiers ..... for discovering the most important biomarkers and using machine learning algorithms (ML) to evaluate the quality of feature sets. 
Predictive models are built using the Random Forest algorithm ([Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324)). It can be applied to two-class problems.

Moreover, EnsembleFS supports users in the analysis and interpretation of molecular data. .

The ensemble-binclass allows the user to:
- filter the most informative features by using up to xxx FS methods (U-test, ) from molecular data generated from high-throughput molecular biology experiments;
- build classifiers using ....
- evaluate the stability of feature subsets (the Lustgarten adjusted stability measure ([ASM](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2815476/)) ) and the performance of predictive models (the area under receiver operator curve  (AUC), accuracy (ACC), and the Matthews correlation coefficient(MCC);
- compare the predictive performance of models and the stability of selected feature sets for selected FS algorithms; 
- establish the selected parameters for predictive models, such as the number of top N informative features;
- remove redundant features by building the Spearman correlation matrix that identifies highly correlated features;
- product plots to visualize the model results;
- find information about selected top molecular markers (gene ontology, pathways, tissue specificity, miRNA targets, regulatory motif, protein complexes, disease phenotypes) in nine biological databases (GO, KEGG, React, WP, TF, MIRNA, HPA, CORUM, and HPO).
