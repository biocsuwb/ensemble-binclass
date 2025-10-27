import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from ReliefF import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, method_: list, size: int,
                 efs: bool = False, efs_method: str = 'union', params: dict = None):
        self.X = X
        self.y = y
        self.efs = efs
        self.efs_method = efs_method
        self.method = method_
        self.size = size
        self.features = None
        self.params = params if params is not None else {}

        if efs:
            self.ranked_features = None
            features_ = []

            for method in method_:
                match method:
                    case 'lasso':
                        features_.append(self.lasso(**self.params))
                    case 'relieff':
                        features_.append(self.relieff(**self.params))
                    case 'mrmr':
                        features_.append(self.mrmr(**self.params))
                    case 'uTest':
                        features_.append(self.u_test())
                    case _:
                        raise ValueError('Unknown method')

            ranking = RankingFeatureSelection(features_, self.size)

            match self.efs_method:
                case 'union':
                    self.ranked_features, self.features = ranking.union()
        else:
            match self.method:
                case 'lasso':
                    self.lasso(**self.params)
                case 'relieff':
                    self.relieff(**self.params)
                case 'mrmr':
                    self.mrmr(**self.params)
                case 'uTest':
                    self.u_test()
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

    def u_test(self):
        class_0 = self.X[self.y == 0]
        class_1 = self.X[self.y == 1]

        p_values = {}
        selected_features = []

        for column in self.X.columns:
            u_statistic, p_value = mannwhitneyu(
                                        class_0[column],
                                        class_1[column],
                                        alternative='two-sided')
            p_values[column] = p_value
            _, p_value_adjusted, _, _ = multipletests(list(p_values.values()), method='fdr_bh')
            selected_features = [column for column, adjusted_p_value in
                                 zip(p_values.keys(), p_value_adjusted) if adjusted_p_value < 0.05]

        self.features = pd.Series(data=selected_features, name="U-TEST")[:self.size]
        return self.features

    def remove_collinear_features(self, threshold: float = 0.75):
        col_corr = set()
        corr_matrix = self.X[self.features].corr("spearman")
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
                    if colname in self.X[self.features].columns:
                        self.X = self.X.drop(colname, axis=1)
                        self.features = self.features[self.features != colname]

    def show_features(self, size: int = 10):
        if size > self.size:
            raise ValueError("size is larger than the list of features")
        print(self.features[:size])


class RankingFeatureSelection(FeatureSelection):
    def __init__(self, features_, size):
        self.features = features_
        self.size = size

    def union(self):
        union = list(set().union(*self.features))

        ranked_features = pd.DataFrame({
            'features': union,
            'importance': np.arange(len(union), 0, -1)
        })

        ensemble_series = pd.Series(union, name='ENSEMBLE')

        return ranked_features, ensemble_series
