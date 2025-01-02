# The ensemble-binclass: an Python package for a feature selection and ensemble classification of molecular omic data
## Project developed by:
Szymon Kołodziejski and Polewko-Klim Aneta
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

![Fig.2](https://github.com/biocsuwb/Images/blob/main/WorkflowEnsembleBin.jpg?raw=true) 

Fig.2 Flowchart of ensemble learning.

## The package consists of the following components:

### Install required packages
To install required packages, run the following command:
```shell
pip install -r ./requirements.txt
```

### Install the development version from GitHub:
To install this package, clone the repository and install with pip:
```python
install.packages("ensemble-binclass")
devtools::install_github("biocsuwb/ensemble-binclass")
```
### Install the development version from PyPi repository:

```python
pip install ensemble-binclass
```

## Import module

```python
from ensbinclass import preprocessing as pre
from ensbinclass import featureSelection as fs
from ensbinclass import classifier as cl
from ensbinclass import ensemble as ens
import pandas as pd
import numpy as np
```
## Example data sets
The RNA-sequencing data of subtypes low-grade glioblastoma (LGG) patients from The Cancer Genome Atlas database ([TCGA](https://www.cancer.gov/tcga)) was used. The preprocessing of data involved standard steps for RNA-Seq data. The log2 transformation was performed. Features with zero and near-zero (1%) variance across patients were removed. After preprocessing, the primary dataset contains xxx tumor samples (wymienic podtypy) described with xxx differentially expressed genes (DEGs). This dataset includes highly correlated features, and the number of cancer samples is roughly ten times more than normal samples. For testing purposes, the number of molecular markers was limited to 2000 DEGs ranked by the highest difference in the gene expression level between subtypes of tumor ([exampleData_TCGA_LUAD_2000.csv](https://github.com/biocsuwb/EnsembleFS-package/tree/main/data)). 
The first column ("class") includes the subtype of patients with LGG. 

## Challenges in analysing molecular data

![Fig.3](https://github.com/biocsuwb/Images/blob/main/Problem_dataomics.jpg?raw=true)

Fig.3 Challenges in analysing molecular data.

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
### Load and check correctness of example data
```python
pr = pre.DataPreprocessing()
pr.load_data('test_data/exampleData_TCGA_LUAD_2000.csv')
pr.show_data()
```
### Prepare omic data for machine learning

#### Choose target column 
```python
X, y = pr.set_target('class')
```

#### Remove collinear features throught the Spearman correlation matrix
```python
pr.remove_collinear_features(threshold=0.75)
```

#### MinMaxScaler features normalization
```python
X = pr.normalization()
```

### Feature selection using the FS methods

#### Feature selection parameters
- X, variables (pd.DataFrame) ***X=X***;
- y, target variable (pd.Series) ***y=y***;
- method_, FS method name (str) ***method_='lasso'***;
- size, top n variables (int) ***size=100***;

#### LASSO hyperparameters 
- params, feature selection method hyperparameters (dict) ***params={'alpha': 0.00001, 'fit_intercept: True, 'precompute': False, 'max_iter': 10000, 'tol': 0.0001, 'selection': 'cyclic', 'random_state': None}***;

#### Run LASSO 
```python
lasso_features = fs.FeatureSelection(X, y, method_='lasso', size=100, params={'alpha': 0.00001, 'fit_intercept': True},)
```

#### RELIEFF hyperparameters
- params, feature selection method hyperparameters (dict) ***params={'n_neighbors': 100}***;

#### Run RELIEFF 
```python
relieff_features = fs.FeatureSelection(X, y, method_='relieff', size=100, params={'n_neighbors': 100},)
```

#### MRMR hyperparameters
- params, feature selection method hyperparameters (dict) ***params={'relevance': 'f', 'redundancy': 'c', 'denominator': 'mean', 'cat_features': None, 'only_same_domain': False, 'return_scores': False, 'n_jobs': -1, 'show_progress': True}***;

#### Run MRMR
```python
mrmr_features = fs.FeatureSelection(X, y, method_='mrmr', size=100, params={'relevance': 'f', 'redundancy': 'c'},)
```

#### U-TEST hyperparameters
- params, feature selection method hyperparameters (dict) ***params={'use_continuity': True, 'alternative': 'two-sided', 'axis': 0, 'method': 'auto'}***;

#### Run U-TEST 
```python
utest_features = fs.FeatureSelection(X, y, method_='uTest', size=100, params={'use_continuity': True, 'alternative': 'two-sided'},)
```

### g:Profiler - interoperable web service for functional enrichment analysis and gene identifier mapping

#### get_profiler configurations parameters
- return_dataframe, result type (bool) if True, query results are presented as pandas DataFrames, otherwise as a list of dicts;
- organism, Organism id for profiling (str) For full list see https://biit.cs.ut.ee/gprofiler/page/organism-list
- query, list of genes to profile (pd.Series or  list)

```python
fs.get_profiler(
  return_dataframe=True,
  organism='hsapiens',
  query=lasso_features[:5],
)
```

|    | source   | native     | name                                             |   p_value | significant   | description                                                                                                                                                                                                                                                                                                                                                                                                                     |   term_size |   query_size |   intersection_size |   effective_domain_size |   precision |    recall | query   | parents                                                  |
|---:|:---------|:-----------|:-------------------------------------------------|----------:|:--------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|-------------:|--------------------:|------------------------:|------------:|----------:|:--------|:---------------------------------------------------------|
|  0 | GO:BP    | GO:0050805 | negative regulation of synaptic transmission     | 0.0267705 | True          | "Any process that stops, prevents, or reduces the frequency, rate or extent of synaptic transmission, the process of communication from a neuron to a target (neuron, muscle, or secretory cell) across a synapse." [GOC:ai]                                                                                                                                                                                                    |          55 |            4 |                   2 |                   21031 |         0.5 | 0.0363636 | query_1 | ['GO:0007268', 'GO:0010648', 'GO:0023057', 'GO:0050804'] |
|  1 | GO:MF    | GO:0019811 | cocaine binding                                  | 0.0498746 | True          | "Binding to cocaine (2-beta-carbomethoxy-3-beta-benzoxytropane), an alkaloid obtained from dried leaves of the South American shrub Erythroxylon coca or by chemical synthesis." [GOC:jl, ISBN:0198506732]                                                                                                                                                                                                                      |           1 |            5 |                   1 |                   20212 |         0.2 | 1         | query_1 | ['GO:0043169', 'GO:0097159', 'GO:1901363']               |
|  2 | GO:MF    | GO:0050785 | advanced glycation end-product receptor activity | 0.0498746 | True          | "Combining with advanced glycation end-products and transmitting the signal to initiate a change in cell activity. Advanced glycation end-products (AGEs) form from a series of chemical reactions after an initial glycation event (a non-enzymatic reaction between reducing sugars and free amino groups of proteins)." [GOC:signaling, PMID:12453678, PMID:12707408, PMID:7592757, PMID:9224812, Wikipedia:RAGE_(receptor)] |           1 |            5 |                   1 |                   20212 |         0.2 | 1         | query_1 | ['GO:0038023']                                           |
 
Table 1. The g:Profiler results for the top 5 features selected by the LASSO method.

## Example 2 - Construct the predictive model with molecular data by using basic classifiers

### Required parameters
- X, variables (pd.DataFrame) ***X=X***;
- y, target (pd.Series) ***y=y***;
- features, list of feature sets (list) ***features=[relieff_features.features, lasso_features.features,]***;
- classifiers, list of classifiers (list) ***classifiers=['adaboost', 'random_forest', 'svm',]***;
- cv, cross-validation method (str) ***cv='stratified_k_fold'***;
- repetitions, number of repetitions (int) ***repetitions=10***;

### Optional parameters
- classifier_params, classifiers hyperparameters (list of dicts) ***classifier_params=[{'adaboost': {'n_estimators': 100, 'learning_rate': 0.9,}}, {'random_forest': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,}}, {'svm': {'C': 1, 'kernel': 'linear', 'gamma': 'auto'}},]***;
- cv_params, cross-validation method hyperparameters (dict) ***cv_params={'n_splits': 10}***;

### Construct the predictive model with molecular data by using basic classifiers

```python
clf = cl.Classifier(
    X,
    y,
    features=[relieff_features.features, lasso_features.features,],
    classifiers=['svm', 'adaboost', 'random_forest',],
    classifier_params=[
        {'svm': {'C': 1, 'kernel': 'linear', 'gamma': 'auto'}},
        {'adaboost': {'n_estimators': 100, 'learning_rate': 0.9}},
        {'random_forest': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None}},
    ],
    cv='stratified_k_fold',
    cv_params={'n_splits': 10},
    repetitions=10,
) 
```

#### Get model metrics
```python
clf.all_metrics()
```

```console
["ACC: {'RELIEFF_SVM': [0.987, 0.015], 'RELIEFF_ADABOOST': [0.994, 0.011], 'RELIEFF_RANDOM_FOREST': [0.99, 0.013], 'LASSO_SVM': [0.995, 0.009], 'LASSO_ADABOOST': [0.992, 0.012], 'LASSO_RANDOM_FOREST': [0.99, 0.014]}",
 "Roc Auc: {'RELIEFF_SVM': [0.978, 0.038], 'RELIEFF_ADABOOST': [0.991, 0.023], 'RELIEFF_RANDOM_FOREST': [0.971, 0.048], 'LASSO_SVM': [0.99, 0.026], 'LASSO_ADABOOST': [0.982, 0.033], 'LASSO_RANDOM_FOREST': [0.971, 0.049]}",
 "F1 score: {'RELIEFF_SVM': [0.993, 0.008], 'RELIEFF_ADABOOST': [0.996, 0.006], 'RELIEFF_RANDOM_FOREST': [0.994, 0.007], 'LASSO_SVM': [0.997, 0.005], 'LASSO_ADABOOST': [0.996, 0.007], 'LASSO_RANDOM_FOREST': [0.994, 0.008]}",
 "MCC: {'RELIEFF_SVM': [0.938, 0.072], 'RELIEFF_ADABOOST': [0.969, 0.051], 'RELIEFF_RANDOM_FOREST': [0.945, 0.07], 'LASSO_SVM': [0.973, 0.047], 'LASSO_ADABOOST': [0.961, 0.06], 'LASSO_RANDOM_FOREST': [0.945, 0.076]}"]
```

#### Visualization of the model metrics

```python
clf.plot_roc_auc()
```

<p align="center">
    <img src="Images/clf_roc_auc.png" alt="Fig.4" width="500">
</p>

Fig. 4 The ROC AUC score for the predictive model with molecular data by using basic classifiers.

## Example 3 - Construct the predictive model with molecular data by using ensemble learning

### Ensemble configuration parameters

### Required parameters
- X, variables (pd.DataFrame) ***X=X***;
- y, target (pd.Series) ***y=y***;
- features, list of feature sets (list) ***features=[relieff_features.features, lasso_features.features,]***;
- classifiers, list of classifiers (list) ***classifiers=['adaboost', 'random_forest', 'svm',]***;
- cv, cross-validation method (str) ***cv='stratified_k_fold'***;
- ensemble, list of ensemble methods (list) ***ensemble=['stacking',]***;
- repetitions, number of repetitions (int) ***repetitions=10***;

### Optional parameters
- classifier_params, classifiers hyperparameters (list of dicts) ***classifier_params=[{'adaboost': {'n_estimators': 100, 'learning_rate': 0.9,}}, {'random_forest': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,}}, {'svm': {'C': 1, 'kernel': 'linear', 'gamma': 'auto'}},]***;
- cv_params, cross-validation method hyperparameters (dict) ***cv_params={'n_splits': 10}***;
- ensemble_params, ensemble hyperparameters (list of dicts) ***ensemble_params=[{'stacking': {'final_estimator': None,}},]***;

### Stacking ensemble learning

```python
ens_stacking = ens.Ensemble(
    X,
    y,
    features=[relieff_features.features, lasso_features.features,],
    classifiers=['adaboost', 'random_forest', 'svm',],
    classifier_params=[
        {'adaboost': {'n_estimators': 100, 'learning_rate': 0.9,}},
        {'random_forest': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,}},
        {'svm': {'C': 1, 'kernel': 'linear', 'gamma': 'auto'}},
   ],  
    cv='stratified_k_fold',
    cv_params={'n_splits': 10},
    ensemble=['stacking',],
    ensemble_params=[
        {'stacking': {'final_estimator': None,}},
    ],
    repetitions=10,
)
```

#### Get model metrics
```python
ens_stacking.all_metrics()
```

```console
["ACC: {'RELIEFF_STACKING': [0.993, 0.01], 'LASSO_STACKING': [0.994, 0.009]}",
 "Roc Auc: {'RELIEFF_STACKING': [0.981, 0.035], 'LASSO_STACKING': [0.984, 0.032]}",
 "F1 score: {'RELIEFF_STACKING': [0.996, 0.006], 'LASSO_STACKING': [0.997, 0.005]}",
 "MCC: {'RELIEFF_STACKING': [0.962, 0.054], 'LASSO_STACKING': [0.968, 0.051]}"]
```

#### Visualization of the model metrics

```python
ens_stacking.plot_roc_auc()
```

<p align="center">
    <img src="Images/ens_stacking_roc_auc.png" alt="Fig.5", width="500">
</p>

Fig. 5 The ROC AUC score for the predictive model with molecular data by using ensemble learning.

### Voting ensemble learning

```python
ens_voting = ens.Ensemble(
    X,
    y,
    features=[relieff_features.features, lasso_features.features,],
    classifiers=['adaboost', 'random_forest', 'svm',],
    classifier_params=[
        {'adaboost': {'n_estimators': 100, 'learning_rate': 0.9,}},
        {'random_forest': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,}},
        {'svm': {'C': 1, 'kernel': 'linear', 'gamma': 'auto'}},
    ],  
    cv='stratified_k_fold',
    cv_params={'n_splits': 10},
    ensemble=['voting',],
    ensemble_params=[
        {'voting': {'voting': 'soft'}},
    ],
    repetitions=10,
)

ens_voting.all_metrics()
```

#### Get model metrics
```python
ens_voting.all_metrics()
```

```console
["ACC: {'RELIEFF_VOTING': [0.991, 0.012], 'LASSO_VOTING': [0.995, 0.009]}",
 "Roc Auc: {'RELIEFF_VOTING': [0.98, 0.039], 'LASSO_VOTING': [0.989, 0.027]}",
 "F1 score: {'RELIEFF_VOTING': [0.995, 0.007], 'LASSO_VOTING': [0.997, 0.005]}",
 "MCC: {'RELIEFF_VOTING': [0.952, 0.061], 'LASSO_VOTING': [0.973, 0.047]}"]
```

#### Visualization of the model metrics

```python
ens_voting.plot_roc_auc()
```

<p align="center">
    <img src="Images/ens_voting_roc_auc.png" alt="Fig.6" width="500">
</p>

Fig. 6 The ROC AUC score for the predictive model with molecular data by using ensemble learning.

### Bagging ensemble learning
```python
ens_bagging = ens.Ensemble(
    X,
    y,
    features=[relieff_features.features, lasso_features.features,],
    classifiers=['adaboost', 'random_forest', 'svm',],
    classifier_params=[
        {'adaboost': {'n_estimators': 100, 'learning_rate': 0.9,}},
        {'random_forest': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None,}},
        {'svm': {'C': 1, 'kernel': 'linear', 'gamma': 'auto'}},
    ],  
    cv='stratified_k_fold',
    cv_params={'n_splits': 10},
    ensemble=[
        'bagging',
    ],
    ensemble_params=[
        {'bagging': {
            'estimator_name': 'random_forest', 'n_estimators': 100, 'max_samples': 0.5, 'max_features': 0.5}},
    ],
    repetitions=10,
)

ens_bagging.all_metrics()
```

#### Get model metrics
```python
ens_bagging.all_metrics()
```

```console
["ACC: {'RELIEFF_BAGGING': [0.989, 0.013], 'LASSO_BAGGING': [0.987, 0.014]}",
 "Roc Auc: {'RELIEFF_BAGGING': [0.961, 0.052], 'LASSO_BAGGING': [0.962, 0.049]}",
 "F1 score: {'RELIEFF_BAGGING': [0.994, 0.007], 'LASSO_BAGGING': [0.993, 0.008]}",
 "MCC: {'RELIEFF_BAGGING': [0.94, 0.069], 'LASSO_BAGGING': [0.931, 0.072]}"]
 ```

#### Visualization of the model metrics

```python
ens_bagging.plot_roc_auc()
```

<p align="center">
    <img src="Images/ens_bagging_roc_auc.png" alt="Fig.7" width="500">
</p>

Fig. 7 The ROC AUC score for the predictive model with molecular data by using ensemble learning.
