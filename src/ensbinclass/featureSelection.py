import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from ReliefF import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, method_: str, size: int, params: dict = None):
        self.X = X
        self.y = y
        self.method = method_
        self.size = size
        self.features = None
        self.params = params if params is not None else {}

        match self.method:
            case 'lasso':
                self.lasso(**self.params)
            case 'relieff':
                self.relieff(**self.params)
            case 'mrmr':
                self.mrmr(**self.params)
            case 'uTest':
                self.u_test(**self.params)
            case _:
                raise ValueError('Unknown method')

    def lasso(self, **kwargs):

        lasso = Lasso(
            alpha=kwargs.get('alpha', 0.00001),
            fit_intercept=kwargs.get('fit_intercept', True),
            precompute=kwargs.get('precompute', False),
            max_iter=kwargs.get('max_iter', 10000),
            tol=kwargs.get('tol', 0.0001),
            selection=kwargs.get('selection', 'cyclic'),
            random_state=kwargs.get('random_state', None),
        )
        lasso.fit(self.X, self.y)
        self.features = pd.Series(data=list(np.array(self.X.columns)[:self.size]), name="LASSO")
        return self.features

    def relieff(self, **kwargs):
        X_array = self.X.values
        y_array = self.y.values

        fs = ReliefF(
            n_neighbors=kwargs.get('n_neighbors', 100),
            n_features_to_keep=kwargs.get('n_features_to_keep', self.size),
        )
        fs.fit(X_array, y_array)

        feature_scores = fs.feature_scores
        feature_scores_df = pd.DataFrame({'Feature': self.X.columns, 'Score': feature_scores})
        top_k_features = feature_scores_df.sort_values(by='Score', ascending=False).head(self.size)
        relieff_features = top_k_features['Feature'].tolist()
        self.features = pd.Series(data=relieff_features, name="RELIEFF")
        return self.features

    def mrmr(self, **kwargs):
        mrmr_features = mrmr_classif(
            self.X,
            self.y,
            K=self.size,
            relevance=kwargs.get('relevance', 'f'),
            redundancy=kwargs.get('redundancy', 'c'),
            denominator=kwargs.get('denominator', 'mean'),
            cat_features=kwargs.get('cat_features', None),
            only_same_domain=kwargs.get('only_same_domain', False),
            return_scores=kwargs.get('return_scores', False),
            n_jobs=kwargs.get('n_jobs', -1),
            show_progress=kwargs.get('show_progress', True),
        )
        self.features = pd.Series(data=mrmr_features, name="MRMR")
        return self.features

    def u_test(self, **kwargs):
        data_class1 = self.y
        data_class2 = self.X

        alpha = 0.05

        p_value_df = pd.DataFrame(index=self.X.columns[:-1], columns=['p_value'])

        def do_utest(i):
            stat, p_value = mannwhitneyu(
                data_class1,
                data_class2.iloc[:, i],
                use_continuity=kwargs.get('use_continuity', True),
                alternative=kwargs.get('alternative', 'two-sided'),
                axis=kwargs.get('axis', 0),
                method=kwargs.get('method', 'auto'),
            )
            p_value_df.loc[self.X.columns[i - 1], 'p_value'] = p_value

        n_features = data_class2.shape[1]

        [do_utest(i) for i in range(n_features)]

        sorted_p_value_df = p_value_df.sort_values(by='p_value', ascending=True)

        utest_features = sorted_p_value_df.loc[sorted_p_value_df['p_value'] < alpha]
        utest_features = utest_features.index.tolist()[:self.size]
        self.features = pd.Series(data=utest_features, name="U-TEST")
        return self.features

    def show_features(self, size: int = 10):
        if size > self.size:
            raise ValueError("size is larger than the list of features")
        print(self.features[:size])
