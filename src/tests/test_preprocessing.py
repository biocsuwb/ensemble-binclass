import numpy as np
import src.preprocessing as preprocessing

pr = preprocessing.DataPreprocessing()
pr.load_data('../../test_data/exampleData_TCGA_LUAD_2000.csv')


def test_load_data() -> None:
    assert pr.data is not None


def test_set_target() -> None:
    X, y = pr.set_target('class')

    assert len(X.shape) == 2
    assert len(y.shape) == 1


def test_one_hot_encoder() -> None:
    cols = pr.data.select_dtypes(exclude=['number']).columns
    for col in cols:
        pr.one_hot_encoder(col)

    assert pr.data.select_dtypes(exclude=['number']).columns.empty


def test_standardization() -> None:
    pr.standardization()

    assert np.all((pr.X >= 0) & (pr.X <= 1.01))
