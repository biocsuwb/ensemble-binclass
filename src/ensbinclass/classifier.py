import time
import numpy as np
import pandas as pd

from ensbinclass.performanceMetrics import PerformanceMetrics
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Classifier:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None,
                 train_index: list = None, test_index: list = None,
                 df_features: list = None, classifiers_w_params: list = None):
        self.X = X
        self.y = y
        self.train_index = train_index
        self.test_index = test_index
        self.features = df_features
        self.classifiers_w_params = classifiers_w_params
        self.classifiers = []
        self.predictions = {}
        self.time = {}
        self.y_true = {}

        for i, feature in enumerate(self.features['fs_method']):
            for fold in range(len(self.train_index)):
                for classifier_w_params in self.classifiers_w_params:
                    for classifier, params in classifier_w_params.items():
                        self.classifiers.append(classifier)
                        match classifier:
                            case 'adaboost':
                                self.predictions[f'ADABOOST-{i}-{feature}-{fold}'] = self.adaboost(fold, i, **params)
                                self.y_true[f'ADABOOST-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'gradient_boosting':
                                self.predictions[f'GRADIENT_BOOSTING-{i}-{feature}-{fold}'] = self.gradient_boosting(fold, i, **params)
                                self.y_true[f'GRADIENT_BOOSTING-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'random_forest':
                                self.predictions[f'RANDOM_FOREST-{i}-{feature}-{fold}'] = self.random_forest(fold, i, **params)
                                self.y_true[f'RANDOM_FOREST-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'k_nearest_neighbors':
                                self.predictions[f'K_NEAREST_NEIGHBORS-{i}-{feature}-{fold}'] = self.k_nearest_neighbors(fold, i, **params)
                                self.y_true[f'K_NEAREST_NEIGHBORS-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'decision_tree':
                                self.predictions[f'DECISION_TREE-{i}-{feature}-{fold}'] = self.decision_tree(fold, i, **params)
                                self.y_true[f'DECISION_TREE-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'extra_trees':
                                self.predictions[f'EXTRA_TREES-{i}-{feature}-{fold}'] = self.extra_trees(fold, i, **params)
                                self.y_true[f'EXTRA_TREES-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'svm':
                                self.predictions[f'SVM-{i}-{feature}-{fold}'] = self.svm(fold, i, **params)
                                self.y_true[f'SVM-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'xgboost':
                                self.predictions[f'XGBOOST-{i}-{feature}-{fold}'] = self.xgb(fold, i, **params)
                                self.y_true[f'XGBOOST-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case 'all':
                                self.predictions[f'ADABOOST-{i}-{feature}-{fold}'] = self.adaboost(fold, i, **params)
                                self.y_true[f'ADABOOST-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'GRADIENT_BOOSTING-{i}-{feature}-{fold}'] = self.gradient_boosting(fold, i, **params)
                                self.y_true[f'GRADIENT_BOOSTING-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'RANDOM_FOREST-{i}-{feature}-{fold}'] = self.random_forest(fold, i, **params)
                                self.y_true[f'RANDOM_FOREST-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'K_NEAREST_NEIGHBORS-{i}-{feature}-{fold}'] = self.k_nearest_neighbors(fold, i, **params)
                                self.y_true[f'K_NEAREST_NEIGHBORS-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'DECISION_TREE-{i}-{feature}-{fold}'] = self.decision_tree(fold, i, **params)
                                self.y_true[f'DECISION_TREE-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'EXTRA_TREES-{i}-{feature}-{fold}'] = self.extra_trees(fold, i, **params)
                                self.y_true[f'EXTRA_TREES-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'SVM-{i}-{feature}-{fold}'] = self.svm(fold, i, **params)
                                self.y_true[f'SVM-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                                self.predictions[f'XGBOOST-{i}-{feature}-{fold}'] = self.xgb(fold, i, **params)
                                self.y_true[f'XGBOOST-{i}-{feature}-{fold}'] = self.y[self.test_index[fold]]
                            case _:
                                raise ValueError('Invalid classifier name')

    def adaboost(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        adaboostClf = AdaBoostClassifier(
            estimator=kwargs.get('estimator_', None),
            n_estimators=kwargs.get('n_estimators', 50),
            learning_rate=kwargs.get('learning_rate', 1.0),
            algorithm=kwargs.get('algorithm', 'SAMME'),
            random_state=kwargs.get('random_state', None),
        )

        adaboostClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )
        predict_proba.append(adaboostClf.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['adaboost'] = end_time - start_time

        return predict_proba

    def gradient_boosting(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        gboostClf = GradientBoostingClassifier(
            loss=kwargs.get('loss', 'log_loss'),
            learning_rate=kwargs.get('learning_rate', 0.1),
            n_estimators=kwargs.get('n_estimators', 100),
            subsample=kwargs.get('subsample', 1.0),
            criterion=kwargs.get('criterion', 'friedman_mse'),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
            max_depth=kwargs.get('max_depth', 3),
            min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
            init=kwargs.get('init', None),
            random_state=kwargs.get('random_state', None),
            max_features=kwargs.get('max_features', None),
            verbose=kwargs.get('verbose', 0),
            max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
            warm_start=kwargs.get('warm_start', False),
            validation_fraction=kwargs.get('validation_fraction', 0.1),
            n_iter_no_change=kwargs.get('n_iter_no_change', None),
            tol=kwargs.get('tol', 1e-4),
            ccp_alpha=kwargs.get('ccp_alpha', 0.0),
        )
        gboostClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )
        predict_proba.append(gboostClf.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['gradient boosting'] = end_time - start_time

        return predict_proba

    def random_forest(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        randomForestClf = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            criterion=kwargs.get('criterion', 'gini'),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
            max_features=kwargs.get('max_features', 'sqrt'),
            max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
            min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
            bootstrap=kwargs.get('bootstrap', True),
            oob_score=kwargs.get('oob_score', False),
            n_jobs=kwargs.get('n_jobs', None),
            random_state=kwargs.get('random_state', None),
            verbose=kwargs.get('verbose', 0),
            warm_start=kwargs.get('warm_start', False),
            class_weight=kwargs.get('class_weight', None),
            ccp_alpha=kwargs.get('ccp_alpha', 0.0),
            max_samples=kwargs.get('max_samples', None),
            monotonic_cst=kwargs.get('monotonic_cst', None),
        )
        randomForestClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )
        predict_proba.append(randomForestClf.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['random forest'] = end_time - start_time

        return predict_proba

    def k_nearest_neighbors(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        kneighborsClf = KNeighborsClassifier(
            n_neighbors=kwargs.get('n_neighbors', 5),
            weights=kwargs.get('weights', 'uniform'),
            algorithm=kwargs.get('algorithm', 'auto'),
            leaf_size=kwargs.get('leaf_size', 30),
            p=kwargs.get('p', 2),
            metric=kwargs.get('metric', 'minkowski'),
            metric_params=kwargs.get('metric_params', None),
            n_jobs=kwargs.get('n_jobs', None),
        )
        kneighborsClf.fit(
            self.X.loc[self.train_index[fold], self.features[fold]]
        )
        predict_proba.append(kneighborsClf.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['k nearest neighbors'] = end_time - start_time

        return predict_proba

    def decision_tree(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        dtreeClf = DecisionTreeClassifier(
            criterion=kwargs.get('criterion', 'gini'),
            splitter=kwargs.get('splitter', 'best'),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
            max_features=kwargs.get('max_features', None),
            random_state=kwargs.get('random_state', None),
            max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
            min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
            class_weight=kwargs.get('class_weight', None),
            ccp_alpha=kwargs.get('ccp_alpha', 0.0),
            monotonic_cst=kwargs.get('monotonic_cst', None),
        )
        dtreeClf_f = dtreeClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )
        predict_proba.append(dtreeClf_f.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['decision tree'] = end_time - start_time

        return predict_proba

    def extra_trees(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        extraTreeClf = ExtraTreesClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            criterion=kwargs.get('criterion', 'gini'),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
            max_features=kwargs.get('max_features', 'sqrt'),
            max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
            min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
            bootstrap=kwargs.get('bootstrap', False),
            oob_score=kwargs.get('oob_score', False),
            n_jobs=kwargs.get('n_jobs', None),
            random_state=kwargs.get('random_state', None),
            verbose=kwargs.get('verbose', 0),
            warm_start=kwargs.get('warm_start', False),
            class_weight=kwargs.get('class_weight', None),
            ccp_alpha=kwargs.get('ccp_alpha', 0.0),
            max_samples=kwargs.get('max_samples', None),
            monotonic_cst=kwargs.get('monotonic_cst', None),
        )
        extraTreeClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )

        end_time = time.time()
        self.time['extra trees'] = end_time - start_time

        return predict_proba

    def svm(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        svmClf = SVC(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            degree=kwargs.get('degree', 3),
            gamma=kwargs.get('gamma', 'scale'),
            coef0=kwargs.get('coef0', 0.0),
            shrinking=kwargs.get('shrinking', True),
            probability=kwargs.get('probability', True),
            tol=kwargs.get('tol', 1e-3),
            cache_size=kwargs.get('cache_size', 200),
            class_weight=kwargs.get('class_weight', None),
            verbose=kwargs.get('verbose', False),
            max_iter=kwargs.get('max_iter', -1),
            decision_function_shape=kwargs.get('decision_function_shape', 'ovr'),
            break_ties=kwargs.get('break_ties', False),
            random_state=kwargs.get('random_state', None),
        )
        svmClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )
        predict_proba.append(svmClf.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['svm'] = end_time - start_time

        return predict_proba

    def xgb(self, fold, i, **kwargs):
        start_time = time.time()

        predict_proba = []
        xgbClf = XGBClassifier(
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.1),
            n_estimators=kwargs.get('n_estimators', 100),
            silent=kwargs.get('silent', True),
            objective=kwargs.get('objective', 'binary:logistic'),
            booster=kwargs.get('booster', 'gbtree'),
            n_jobs=kwargs.get('n_jobs', 1),
            nthread=kwargs.get('nthread', None),
            gamma=kwargs.get('gamma', 0),
            min_child_weight=kwargs.get('min_child_weight', 1),
            max_delta_step=kwargs.get('max_delta_step', 0),
            subsample=kwargs.get('subsample', 1),
            colsample_bytree=kwargs.get('colsample_bytree', 1),
            colsample_bylevel=kwargs.get('colsample_bylevel', 1),
            reg_alpha=kwargs.get('reg_alpha', 0),
            reg_lambda=kwargs.get('reg_lambda', 1),
            scale_pos_weight=kwargs.get('scale_pos_weight', 1),
            base_score=kwargs.get('base_score', 0.5),
            random_state=kwargs.get('random_state', None),
            seed=kwargs.get('seed', None),
            missing=kwargs.get('missing', np.nan),
        )
        xgbClf.fit(
            self.X.loc[self.train_index[fold], self.features['features'][i]],
            self.y[self.train_index[fold]]
        )
        predict_proba.append(xgbClf.predict_proba(
            self.X.loc[self.test_index[fold], self.features['features'][i]]
        ))

        end_time = time.time()
        self.time['xgb'] = end_time - start_time

        return predict_proba

    def accuracy(self):
        pm = PerformanceMetrics(self)
        return pm.accuracy()

    def roc_auc(self):
        pm = PerformanceMetrics(self)
        return pm.roc_auc()

    def f1_score(self):
        pm = PerformanceMetrics(self)
        return pm.f1_score()

    def matthews_corrcoef(self):
        pm = PerformanceMetrics(self)
        return pm.matthews_corrcoef()

    def precision(self):
        pm = PerformanceMetrics(self)
        return pm.precision()

    def recall(self):
        pm = PerformanceMetrics(self)
        return pm.recall()

    def get_metrics(self):
        pm = PerformanceMetrics(self)
        return pm.get_metrics()

    def std(self, X):
        pm = PerformanceMetrics(self)
        return pm.std(X)

    def all_metrics(self):
        pm = PerformanceMetrics(self)
        return pm.all_metrics()

    def plot_acc(self):
        pm = PerformanceMetrics(self)
        pm.plot_acc()

    def plot_roc_auc(self):
        pm = PerformanceMetrics(self)
        pm.plot_roc_auc()

    def plot_f1_score(self):
        pm = PerformanceMetrics(self)
        pm.plot_f1_score()

    def plot_mcc(self):
        pm = PerformanceMetrics(self)
        pm.plot_mcc()

    def plot_all(self):
        pm = PerformanceMetrics(self)
        pm.plot_all()
