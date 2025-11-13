import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.metrics import jaccard_score
from skrebate import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, train_index: list = None,
                 methods_w_params: list = None, size: int = 100):
        self.X = X
        self.y = y
        self.train_index = train_index
        if self.train_index is None:
            self.train_index = [np.array(self.X.index)]
        self.size = size
        self.features = None
        self.methods_w_params = methods_w_params
        self.feature_importance_list = []
        self.feature_importance_df = None
        self.feature_stability = None
        self.iterate = 0

        for method_w_params in self.methods_w_params:
            for method, params in method_w_params.items():
                match method:
                    case 'lasso':
                        self.lasso(**params)
                    case 'relieff':
                        self.relieff(**params)
                    case 'mrmr':
                        self.mrmr(**params)
                    case 'uTest':
                        self.u_test(**params)

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
            self.iterate += 1

            features = pd.DataFrame({
                'fs_method': 'LASSO',
                'features': selected_features,
                'importance': top_coefs.values,
                'rank': list(range(1, len(selected_features) + 1)),
                'fold': i,
                'iter': self.iterate,
            })

            self.feature_importance_list.append(features)

        return self.feature_importance_list

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
            self.iterate += 1

            features = pd.DataFrame({
                'fs_method': 'RELIEFF',
                'features': feature_scores_df['Feature'].tolist(),
                'importance': feature_scores_df['Importance'].tolist(),
                'rank': list(range(1, feature_scores_df.shape[0] + 1)),
                'fold': i,
                'iter': self.iterate,
            })

            self.feature_importance_list.append(features)

        return self.feature_importance_list

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
            self.iterate += 1

            features = pd.DataFrame({
                'fs_method': 'MRMR',
                'features': feature_scores_df['Feature'].tolist(),
                'importance': feature_scores_df['Importance'].tolist(),
                'rank': list(range(1, feature_scores_df.shape[0] + 1)),
                'fold': i,
                'iter': self.iterate,
            })

            self.feature_importance_list.append(features)

        return self.feature_importance_list

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
            feature_p_value_filtered = feature_p_value[
                feature_p_value['Importance'] < kwargs.get('alpha', 0.05)
            ].reset_index(drop=True)
            self.iterate += 1

            features = pd.DataFrame({
                'fs_method': 'UTest',
                'features': feature_p_value_filtered['Feature'].tolist(),
                'importance': feature_p_value_filtered['Importance'].tolist(),
                'rank': list(range(1, feature_p_value_filtered.shape[0] + 1)),
                'fold': i,
                'iter': self.iterate
            })

            self.feature_importance_list.append(features)

        return self.feature_importance_list

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
        df_long = pd.concat(self.feature_importance_list, ignore_index=True)
        self.feature_importance_df = (
            df_long.groupby(['fs_method', 'fold', 'iter'])
            .agg({
                'features': list,
                'importance': list,
                'rank': list
            })
            .reset_index()
        )
        return self.feature_importance_df

    def _nogueira_stability(self, Z):
        M, d = Z.shape
        hatPF = np.mean(Z, axis=0)
        kbar = np.sum(hatPF)
        denom = (kbar / d) * (1 - kbar / d)
        return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom

    def compute_method_stabilities(self):
        if self.feature_importance_df is None:
            self.get_feature_importance()

        stability = []

        for method in self.feature_importance_df['fs_method'].unique():
            df_method = self.feature_importance_df[self.feature_importance_df['fs_method'] == method]

            for fold in df_method['fold'].unique():
                df_fold_subset = df_method[df_method['fold'] <= fold]
                feature_sets = df_fold_subset['features'].tolist()

                if len(feature_sets) < 2:
                    jaccard_stability = np.nan
                    nogueira_stability = np.nan
                else:
                    all_features = list(set(f for subset in feature_sets for f in subset))
                    M = len(feature_sets)
                    p = len(all_features)

                    selection_matrix = np.zeros((M, p), dtype=int)
                    feature_index = {f: i for i, f in enumerate(all_features)}
                    for i, features in enumerate(feature_sets):
                        for f in features:
                            selection_matrix[i, feature_index[f]] = 1

                    jaccards = []
                    for a, b in combinations(range(M), 2):
                        A, B = selection_matrix[a, :], selection_matrix[b, :]
                        jaccards.append(jaccard_score(A, B))

                    jaccard_stability = np.mean(jaccards)

                    nogueira_stability = self._nogueira_stability(selection_matrix)

                stability.append({
                    'fs_method': method,
                    'fold': fold,
                    'jaccard_stability': jaccard_stability,
                    'nogueira_stability': nogueira_stability,
                    'n_sets_used': len(feature_sets)
                })

        stability_df = pd.DataFrame(stability)

        summary_df = (
            stability_df
            .groupby('fs_method', as_index=False)
            .agg({
                'jaccard_stability': 'mean',
                'nogueira_stability': 'mean'
            })
            .assign(fold='mean', n_sets_used='all')
            [['fs_method', 'fold', 'jaccard_stability', 'nogueira_stability', 'n_sets_used']]
        )

        self.feature_stability = pd.concat([stability_df, summary_df], ignore_index=True)

        return self.feature_stability
