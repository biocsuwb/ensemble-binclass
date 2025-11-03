import pandas as pd
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, method_: str, size: int,
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
        self.feature_importance.attrs['name'] = "LASSO"
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
        self.feature_importance.attrs['name'] = "RELIEFF"
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

        mrmr_importances = (mrmr_features[1]).reindex(mrmr_features[0])
        self.feature_importance = pd.DataFrame({
            'Feature': mrmr_importances.index.tolist(),
            'Importance': mrmr_importances.values
        })
        self.feature_importance.attrs['name'] = "MRMR"
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
        self.feature_importance.attrs['name'] = "UTEST"
        self.features = pd.Series(data=self.feature_importance['Feature'], name="UTEST")

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
