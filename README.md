# The ensemble-binclass: an Python package for a feature selection and ensemble classification of molecular omic data
## Description
ensemble-binclass is an python package for single feature selection (FS) and ensemble classification of omic data (numeric data formats).
This tool is based on several feature filters, such as ... and classifiers ..... for discovering the most important biomarkers and using machine learning algorithms (ML) to evaluate the quality of feature sets. 
Predictive models are built using the Random Forest algorithm ([Breiman 2001](https://link.springer.com/article/10.1023/A:1010933404324)). It can be applied to two-class problems.

**Ensemble-binclass is a Python package that allows the user to:**
* filter the most informative features by using up to four FS methods (U-test, MRMR, Relief, LASSO) from numeric omics data generated from high-throughput molecular biology experiments;
* remove redundant features by building the Spearman correlation matrix that identifies highly correlated features;
* perform binary classification by using eight classifiers (Adaboost, gradient boosting, random forest, k-nearest neighbor (k-NN), decision tree, extra trees, support vector machine (SVM) i extreme gradient boost (XGBoost)):
* build predictive model using one of ensemble learning, namely, stacking, boosting, and bagging (see Figure 1);
* evaluate predictive model by using several metrics (the area under receiver operator curve  (AUC), accuracy (ACC), and the Matthews correlation coefficient(MCC), F1, ....);
* compare the predictive performance of models and the stability of selected feature sets for selected FS algorithms, individual classifiers, and combined classifiers;  
* establish the selected parameters for predictive models, such as the number of top N informative features;
* product plots to visualize the model results;
  
![Fig.1](https://github.com/biocsuwb/Images/blob/main/EnsembleLearning.jpg?raw=true) 

Fig.1 Ensemble learning.

## Install ensemble-binclass package
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
