import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
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
        self.feature_importance = None

        if efs:
            self.ranked_features = None
            rank_feature_importance = pd.DataFrame()

            for method in method_:
                match method:
                    case 'lasso':
                        _, lasso_df = self.lasso(**self.params)
                        lasso_df['Method'] = method
                        rank_feature_importance = pd.concat([rank_feature_importance, lasso_df])
                    case 'relieff':
                        _, relieff_df = self.relieff(**self.params)
                        relieff_df['Method'] = method
                        rank_feature_importance = pd.concat([rank_feature_importance, relieff_df])
                    case 'mrmr':
                        _, mrmr_df = self.mrmr(**self.params)
                        mrmr_df['Method'] = method
                        rank_feature_importance = pd.concat([rank_feature_importance, mrmr_df])
                    case 'uTest':
                        _, utest_df = self.u_test()
                        utest_df['Method'] = method
                        rank_feature_importance = pd.concat([rank_feature_importance, utest_df])
                    case _:
                        raise ValueError('Unknown method')

            ranking = RankingFeatureSelection(rank_feature_importance)

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
            positive=kwargs.get('positive', True),
        )
        lasso.fit(self.X, self.y)

        coefs = pd.Series(lasso.coef_, index=self.X.columns)
        top_coefs = coefs.abs().sort_values(ascending=False)[:self.size]
        selected_features = top_coefs.index.tolist()
        self.feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': top_coefs.values})
        self.features = pd.Series(self.feature_importance['Feature'], name='LASSO')

        return self.features, self.feature_importance

    def relieff(self, **kwargs):
        X_array = self.X.values
        y_array = self.y.values

        fs = ReliefF(
            n_neighbors=kwargs.get('n_neighbors', 10),
            n_features_to_select=self.size,
        )
        fs.fit(X_array, y_array)

        feature_scores = fs.feature_importances_
        feature_scores_df = pd.DataFrame({'Feature': self.X.columns, 'Importance': feature_scores})
        feature_scores_df = feature_scores_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        feature_scores_df = feature_scores_df[0:self.size]
        self.feature_importance = feature_scores_df
        self.features = pd.Series(data=self.feature_importance['Feature'], name="RELIEFF").reset_index(drop=True)

        return self.features, self.feature_importance

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
            return_scores=kwargs.get('return_scores', True),
            n_jobs=kwargs.get('n_jobs', -1),
            show_progress=kwargs.get('show_progress', True),
        )

        mrmr_importance = pd.DataFrame({
            'Feature': mrmr_features[1].index.to_list()[:self.size],
            'Importance': mrmr_features[1].values[:self.size],
        })
        self.feature_importance = mrmr_importance.sort_values(by='Importance', ascending=False)
        self.features = pd.Series(data=self.feature_importance['Feature'], name="MRMR").reset_index(drop=True)

        return self.features, self.feature_importance

    def u_test(self, alpha=0.05):
        class_0 = self.X[self.y == 0]
        class_1 = self.X[self.y == 1]

        selected_features = []
        p_values = []

        for column in self.X.columns:
            _, p_value = mannwhitneyu(
                class_0[column],
                class_1[column],
                alternative='two-sided')
            selected_features.append(column)
            p_values.append(p_value)

        feature_p_value = pd.DataFrame({'Feature': selected_features, 'P-value': p_values})
        feature_p_value = feature_p_value.sort_values(by='P-value').reset_index(drop=True)

        _, p_value_adjusted, _, _ = multipletests(feature_p_value['P-value'], method='fdr_bh')

        feature_p_value['Importance'] = p_value_adjusted
        self.feature_importance = (
            feature_p_value[feature_p_value['Importance'] < alpha]
            .head(self.size)
        )
        self.features = pd.Series(data=self.feature_importance['Feature'], name="U-TEST")

        return self.features, self.feature_importance

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

    def get_feature_importance(self):
        return self.feature_importance


class RankingFeatureSelection(FeatureSelection):
    def __init__(self, rank_feature_importance: pd.DataFrame):
        self.rank_feature_importance = rank_feature_importance.copy()
        self.rank_feature_importance['Importance'] = pd.to_numeric(self.rank_feature_importance['Importance'],
                                                                   errors='coerce').fillna(0.0)

    def _normalize_per_method(self) -> pd.DataFrame:
        df = self.rank_feature_importance.copy()

        def transform_group(g):
            method = g['Method'].iloc[0]
            scores = g['Importance'].astype(float).to_numpy()

            if method == 'uTest':
                eps = 1e-300
                clipped = np.clip(scores, eps, 1.0)
                with np.errstate(divide='ignore'):
                    transformed = -np.log10(clipped)
            else:
                transformed = np.array(scores, dtype=float)

            transformed_col = np.asarray(transformed, dtype=float).reshape(-1, 1)

            if transformed_col.size == 0 or np.isclose(transformed_col.min(), transformed_col.max(), rtol=0, atol=0):
                normalized = np.ones_like(transformed_col, dtype=float)
            else:
                scaler = MinMaxScaler(feature_range=(0.0, 1.0))
                normalized = scaler.fit_transform(transformed_col)

            g = g.assign(Importance=normalized.ravel())
            return g

        normalized_df = df.groupby('Method', group_keys=False).apply(transform_group).reset_index(drop=True)
        return normalized_df

    def union(self):
        normalized_df = self._normalize_per_method()

        ranked_features = (
            normalized_df
            .groupby('Feature', sort=False)['Importance']
            .mean()
            .sort_values(ascending=False, kind='mergesort')
        )

        ensemble_series = pd.Series(data=ranked_features.index.to_list(), name='ENSEMBLE')

        return ranked_features, ensemble_series
