import src.preprocessing as preprocessing
import src.featureSelection as featureSelection
import src.classifier as classifier

pr = preprocessing.DataPreprocessing()
pr.load_data('../../test_data/exampleData_TCGA_LUAD_2000.csv')
X, y = pr.set_target('class')
features = featureSelection.FeatureSelection(X, y, 'lasso', 100)
features = features.get_features()


def test_classifiers():
    clf = classifier.Classifier(X, y, features=features, classifiers=['all'],
                                cv='stratified_k_fold', fold=10)

    variables = [clf.X, clf.fs, clf.y, clf.X_train, clf.X_test, clf.y_train,
                 clf.y_test, clf.cross_validation, clf.classifiers,
                 clf.predictions, clf.time, clf.fold]

    assert any(var is None for var in variables) is False
