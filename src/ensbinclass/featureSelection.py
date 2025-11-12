import pandas as pd
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, train_index: list = None,
                 methods: list = None, size: int = 100, params: dict = None):
        self.X = X
        self.y = y
        self.train_index = train_index
        self.size = size
        self.features = None
        self.params = params if params is not None else {}
        self.feature_importance = pd.DataFrame(
            columns=['fs_method', 'features', 'importance', 'rank', 'fold', 'iter']
        )
        self.iter = 0

        for method in methods:
            match method:
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
        for i, train_idx in enumerate(self.train_index):
            lasso = Lasso(
                alpha=kwargs.get('alpha', 0.00001),
                fit_intercept=kwargs.get('fit_intercept', True),
                precompute=kwargs.get('precompute', False),
                copy_X=kwargs.get('copy_X', True),
                max_iter=kwargs.get('max_iter', 1000),
                tol=kwargs.get('tol', 0.0001),
                warm_start=kwargs.get('warm_start', False),
                positive=kwargs.get('positive', True),
                random_state=kwargs.get('random_state', None),
                selection=kwargs.get('selection', 'cyclic'),
            )
            lasso.fit(
                self.X.iloc[train_idx, :],
                self.y.iloc[train_idx]
            )

            coefs = pd.Series(lasso.coef_, index=self.X.columns)
            top_coefs = coefs.abs().sort_values(ascending=False)[:self.size]
            selected_features = top_coefs.index.tolist()
            self.iter += 1

            features = {
                'fs_method': 'LASSO',
                'features': selected_features,
                'importance': top_coefs.values,
                'rank': list(range(1, len(selected_features) + 1)),
                'fold': i,
                'iter': self.iter,
            }

            self.feature_importance.loc[len(self.feature_importance)] = features

        return self.feature_importance

    def relieff(self, **kwargs):
        for i, train_idx in enumerate(self.train_index):
            X_array = self.X.iloc[train_idx, :].values
            y_array = self.y.iloc[train_idx].values

            fs = ReliefF(
                n_neighbors=kwargs.get('n_neighbors', 10),
                n_features_to_select=self.size,
            )
            fs.fit(X_array, y_array)

            feature_score = fs.feature_importances_
            feature_scores_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': feature_score,
            })
            feature_scores_df = feature_scores_df.sort_values(
                by='Importance',
                ascending=False
            ).reset_index(drop=True)
            feature_scores_df = feature_scores_df[0:self.size]
            self.iter += 1

            features = {
                'fs_method': 'RELIEFF',
                'features': feature_scores_df['Feature'].tolist(),
                'importance': feature_scores_df['Importance'].tolist(),
                'rank': list(range(1, feature_scores_df.shape[0] + 1)),
                'fold': i,
                'iter': self.iter,
            }

            self.feature_importance.loc[len(self.feature_importance)] = features

        return self.feature_importance

    def mrmr(self, **kwargs):
        for i, train_idx in enumerate(self.train_index):
            mrmr_features = mrmr_classif(
                self.X.iloc[train_idx, :],
                self.y.iloc[train_idx],
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

            mrmr_importances = mrmr_features[1].loc[mrmr_features[1].index.isin(mrmr_features[0])]
            feature_scores_df = pd.DataFrame({
                'Feature': mrmr_importances.index.tolist(),
                'Importance': mrmr_importances.values
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
            self.iter += 1

            features = {
                'fs_method': 'MRMR',
                'features': feature_scores_df['Feature'].tolist(),
                'importance': feature_scores_df['Importance'].tolist(),
                'rank': list(range(1, feature_scores_df.shape[0] + 1)),
                'fold': i,
                'iter': self.iter,
            }

            self.feature_importance.loc[len(self.feature_importance)] = features

        return self.feature_importance

    def u_test(self, **kwargs):
        for i, train_idx in enumerate(self.train_index):
            X = self.X.iloc[train_idx, :]
            y = self.y.iloc[train_idx]

            class_0 = X[y == 0]
            class_1 = X[y == 1]

            selected_features = []
            p_values = []

            for column in X.columns:
                _, p_value = mannwhitneyu(
                    class_0[column],
                    class_1[column],
                    use_continuity=kwargs.get('use_continuity', True),
                    alternative='two-sided',
                    axis=kwargs.get('axis', 0),
                    method='auto',
                    nan_policy='propagate',
                    keepdims=kwargs.get('keepdims', False),
                )
                selected_features.append(column)
                p_values.append(p_value)

            feature_p_value = pd.DataFrame({'Feature': selected_features, 'P-value': p_values})

            _, p_value_adjusted, _, _ = multipletests(feature_p_value['P-value'], method='fdr_bh')

            feature_p_value = (pd.DataFrame({'Feature': selected_features, 'P-value': p_values})
                               .sort_values(by='P-value', ascending=True)
                               .reset_index(drop=True))
            _, p_value_adjusted, _, _ = multipletests(feature_p_value['P-value'], method='fdr_bh')
            feature_p_value['Importance'] = p_value_adjusted
            feature_p_value_filtered = (feature_p_value[
                feature_p_value['Importance'] < kwargs.get('alpha', 0.05)]
                .head(self.size)
                .reset_index(drop=True))
            self.iter += 1

            features = {
                'fs_method': 'UTest',
                'features': feature_p_value_filtered['Feature'].tolist(),
                'importance': feature_p_value_filtered['Importance'].tolist(),
                'rank': list(range(1, feature_p_value_filtered.shape[0] + 1)),
                'fold': i,
                'iter': self.iter,
            }

            self.feature_importance.loc[len(self.feature_importance)] = features

        return self.feature_importance

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
