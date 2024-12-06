# The ensemble-binclass: an Python package for a feature selection and ensemble classification of molecular omic data
## Project developed by:
Polewko-Klim Aneta and Szymon Kołodziejski
## Description
ensemble-binclass is an python package for feature selection (FS) and ensemble classification of omic data (numeric data formats).
This tool is based on four feature filters (**U-test**, the minimum redundancy maximum relevance (**MRMR**) 
([Ding 2005](https://pubmed.ncbi.nlm.nih.gov/15852500/), the **Relief** ([Kira 1992](https://dl.acm.org/doi/abs/10.5555/1867135.1867155)), the least absolute shrinkage and selection operator (**LASSO**) ([Tibshirani 1996](https://www.jstor.org/stable/2346178))) and eight binary classifiers
(**Adaboost**, **gradient boosting**, **random forest** [Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324), k-nearest neighbor (**k-NN**), **decision tree**, **extra trees**, support vector machine (**SVM**) and extreme gradient boost (**XGBoost**)).
The ensemble-binclass uses the machine learning algorithms (ML) to evaluate the quality of feature sets (**k-fold stratified crossvalidation** or **resampling**). 
It can be applied to two-class problems.

**Ensemble-binclass is a Python package that allows the user to:**
* filter the most informative features by using up to four FS methods from numeric omics data generated from high-throughput molecular biology experiments;
* remove redundant features by building the Spearman correlation matrix that identifies highly correlated features;
* perform binary classification by using one of the eight classifiers:
* build predictive model using one of ensemble learning, namely, **stacking, boosting, and bagging** (see Figure 1);
* evaluate predictive model by using several metrics (the area under receiver operator curve  (AUC), accuracy (ACC), and the Matthews correlation coefficient(MCC), F1, ....);
* compare the predictive performance of models and the stability of selected feature sets for selected FS algorithms, individual classifiers, and combined classifiers;  
* establish the selected parameters for predictive models, such as the number of top N informative features;
* product plots to visualize the model results;
  
![Fig.1](https://github.com/biocsuwb/Images/blob/main/EnsembleLearning.jpg?raw=true) 

Fig.1 The ensemble learning strategy in ensemble-binclass.

## General workflow of the ensemble learning
<p align="center">
    <img src="https://github.com/biocsuwb/Images/blob/main/WorkflowEnsembleBin.jpg?raw=true" alt="Fig.3 Flowchart of ensemble learning. " width="500">
</p>

## Install package
### Install the development version from GitHub:
To install this package, clone the repository and install with pip:
```r
install.packages("ensemble-binclass")
devtools::install_github("biocsuwb/ensemble-binclass")
```
### Install the development version from PyPi repository:
```r
pip install ensemble-binclass
```
## Import module
```r
from ensemble-binclass import ... as ...
from ensemble-binclass import ... as ...
import pandas as pd
import numpy as np
```
## Example data sets
The RNA-sequencing data of subtypes low-grade glioblastoma (LGG) patients from The Cancer Genome Atlas database ([TCGA](https://www.cancer.gov/tcga)) was used. The preprocessing of data involved standard steps for RNA-Seq data. The log2 transformation was performed. Features with zero and near-zero (1%) variance across patients were removed. After preprocessing, the primary dataset contains xxx tumor samples (wymienic podtypy) described with xxx differentially expressed genes (DEGs). This dataset includes highly correlated features, and the number of cancer samples is roughly ten times more than normal samples. For testing purposes, the number of molecular markers was limited to 2000 DEGs ranked by the highest difference in the gene expression level between subtypes of tumor ([exampleData_TCGA_LUAD_2000.csv](https://github.com/biocsuwb/EnsembleFS-package/tree/main/data)). 
The first column ("class") includes the subtype of patients with LGG. 

## Challenges in analysing molecular data
<p align="center">
    <img src="https://github.com/biocsuwb/Images/blob/main/Problem_dataomics.jpg?raw=true" alt="Fig.2" width="650">
</p>

## Detailed step-by-step instruction on how to conduct the analysis:
1. Identify relevant variables (candidate diagnostic biomarkers) by using FS methods
2. Remove redundant variables (correlated biomarkers)
3. Select top-N relevant variables 
4. Select a method to validate the ML model
5. Construct predictive model using individual classifier or ensemble classifier 
6. Evaluate biomarker discovery methods
7. Perform hyperparameter optimization for classifier algorithm
8. Construct diagnosis system using the best classifier 

## Example 1 - Construct the predictive model with molecular data by using one of eight basic classifiers
### Load example training data
```r
download.file("https://raw.githubusercontent.com/biocsuwb/EnsembleFS-package/main/data/correctData/correctData/df.RNA.merge.image.LGG.csv", 
              destfile = "correctData/correctData/df.RNA.merge.image.LGG.csv", method = "curl")

data_RNA = dp.OmicDataPreprocessing(path='correctData/df.RNA.merge.image.LGG.csv')

data_RNA.load_data()
```
### Prepare omic data for machine learning
```r
```
#### Model (optional) configuration parameters
- U-test parameter, multitest correction:  ***adjust = {"holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"}***;
- U-test parameter, significance level: ***alpha = 0.05***;
- validation methods: ***method.cv = {'kfoldcv','rsampling'}***;
- number of repetitions: ***niter = 10***;
- train-test-split the data: ***k = 3*** for stratified k-fold cross-validation and ***test.size = 0.3*** for random sampling;
- classifier: SVM;
- classifier parameter: ***kernel = ***{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}***;
## Example 2 - Construct the predictive model with molecular data by using stacking ensemble learning
```r
```
## Example 3 - Construct the predictive model with molecular data by using boosting ensemble learning
```r
```
## Example 4 - Construct the predictive model with molecular data by using bagging ensemble learning
```r
```
