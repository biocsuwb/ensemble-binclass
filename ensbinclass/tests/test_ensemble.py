import numpy as np

import ensbinclass.preprocessing as preprocessing
import ensbinclass.ensemble as ensemble
import ensbinclass.featureSelection as featureSelection

np.seterr(divide='ignore')

pr = preprocessing.DataPreprocessing()
pr.load_data('../../test_data/exampleData_TCGA_LUAD_2000.csv')
X, y = pr.set_target('class')
features = featureSelection.FeatureSelection(X, y, 'lasso', 100)
features = features.get_features()


def test_voting():
    ens = ensemble.Ensemble(X, y, features=features, ensemble=['voting'],
                            classifiers=['adaboost', 'random_forest', 'xgb'],
                            cross_validation='stratified_k_fold', fold=10)

    variables = [ens.X, ens.fs, ens.y, ens.X_train, ens.X_test, ens.y_train,
                 ens.y_test, ens.ensemble, ens.cross_validation, ens.classifiers,
                 ens.model_classifiers, ens.predictions, ens.n_splits, ens.time]

    assert any(var is None for var in variables) is False


def test_bagging():
    ens = ensemble.Ensemble(X, y, features=features, ensemble=['bagging'],
                            classifiers=['adaboost', 'random_forest', 'xgb'],
                            cross_validation='stratified_k_fold', fold=10)

    variables = [ens.X, ens.fs, ens.y, ens.X_train, ens.X_test, ens.y_train,
                 ens.y_test, ens.ensemble, ens.cross_validation, ens.classifiers,
                 ens.model_classifiers, ens.predictions, ens.n_splits, ens.time]

    assert any(var is None for var in variables) is False


def test_stacking():
    ens = ensemble.Ensemble(X, y, features=features, ensemble=['stacking'],
                            classifiers=['adaboost', 'random_forest', 'xgb'],
                            cross_validation='stratified_k_fold', fold=10)

    variables = [ens.X, ens.fs, ens.y, ens.X_train, ens.X_test, ens.y_train,
                 ens.y_test, ens.ensemble, ens.cross_validation, ens.classifiers,
                 ens.model_classifiers, ens.predictions, ens.n_splits, ens.time]

    assert any(var is None for var in variables) is False
