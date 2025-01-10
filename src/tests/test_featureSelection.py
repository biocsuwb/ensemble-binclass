import pytest
import src.ensbinclass.featureSelection as featureSelection
import src.ensbinclass.preprocessing as preprocessing

NUM_FEATURES = 10

pr = preprocessing.DataPreprocessing()
pr.load_data('../../test_data/exampleData_TCGA_LUAD_2000.csv')
X, y = pr.set_target('class')


def test_lasso_fs():
    lasso_fs = featureSelection.FeatureSelection(X, y, 'lasso', NUM_FEATURES)
    lasso_fs = lasso_fs.get_features()
    assert len(lasso_fs) == NUM_FEATURES


def test_relieff_fs():
    relieff_fs = featureSelection.FeatureSelection(X, y, 'relieff', NUM_FEATURES)
    relieff_fs = relieff_fs.get_features()
    assert len(relieff_fs) == NUM_FEATURES


def test_mrmr_fs():
    mrmr_fs = featureSelection.FeatureSelection(X, y, 'mrmr', NUM_FEATURES)
    mrmr_fs = mrmr_fs.get_features()
    assert len(mrmr_fs) == NUM_FEATURES


def test_uTest_fs():
    uTest_fs = featureSelection.FeatureSelection(X, y, 'uTest', NUM_FEATURES)
    uTest_fs = uTest_fs.get_features()
    assert len(uTest_fs) == NUM_FEATURES


def test_unknown_method():
    with pytest.raises(ValueError):
        unknown_fs = featureSelection.FeatureSelection(X, y, 'unknown', NUM_FEATURES)
        unknown_fs.get_features()
