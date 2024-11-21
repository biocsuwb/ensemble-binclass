import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from ReliefF import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, method: str, size: int, **kwargs):
        self.X = X
        self.y = y
        self.method = method
        self.size = size
        self.features = None

        match self.method:
            case 'lasso':
                self.lasso(**kwargs)
            case 'relieff':
                self.relieff(**kwargs)
            case 'mrmr':
                self.mrmr()
            case 'uTest':
                self.u_test()
            case _:
                raise ValueError('Unknown method')

    def lasso(self, **kwargs):
        alpha = kwargs.get('alpha', 0.00001)
        max_iter = kwargs.get('max_iter', 10000)
        lasso = Lasso(alpha=alpha, max_iter=max_iter)
        lasso.fit(self.X, self.y)
        self.features = pd.Series(data=list(np.array(self.X.columns)[:self.size]), name="Lasso")
        return self.features

    def relieff(self, **kwargs):
        n_features_to_keep = kwargs.get('n_features_to_keep', self.size)
        X_array = self.X.values
        y_array = self.y.values

        fs = ReliefF(n_neighbors=100, n_features_to_keep=n_features_to_keep)
        fs.fit(X_array, y_array)

        feature_scores = fs.feature_scores
        feature_scores_df = pd.DataFrame({'Feature': self.X.columns, 'Score': feature_scores})
        top_k_features = feature_scores_df.sort_values(by='Score', ascending=False).head(self.size)
        relieff_features = top_k_features['Feature'].tolist()
        self.features = pd.Series(data=relieff_features, name="ReliefF")
        return relieff_features

    def mrmr(self):
        mrmr_features = mrmr_classif(self.X, self.y, K=self.size)
        self.features = pd.Series(data=mrmr_features, name="Mrmr")
        return mrmr_features

    def u_test(self):
        data_class1 = self.y
        data_class2 = self.X

        alpha = 0.05

        p_value_df = pd.DataFrame(index=self.X.columns[:-1], columns=['p_value'])

        def do_utest(i):
            stat, p_value = mannwhitneyu(data_class1, data_class2.iloc[:, i])
            p_value_df.loc[self.X.columns[i - 1], 'p_value'] = p_value

        n_features = data_class2.shape[1]

        [do_utest(i) for i in range(n_features)]

        sorted_p_value_df = p_value_df.sort_values(by='p_value', ascending=True)

        utest_features = sorted_p_value_df.loc[sorted_p_value_df['p_value'] < alpha]
        utest_features = utest_features.index.tolist()[:self.size]
        self.features = pd.Series(data=utest_features, name="U-test")
        return utest_features

    def show_features(self, size: int = 10):
        if size > self.size:
            raise ValueError("size is larger than the list of features")
        print(self.features[:size])

    def get_features(self):
        return self.features
