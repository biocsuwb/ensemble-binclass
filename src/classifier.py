import time
import pandas as pd
import src.modelEvaluation as modelEvaluation
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Classifier:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None, features: pd.Series = None,
                 classifiers: list = None, classifier_params: dict = None,
                 cv: str = 'hold_out', cv_params: dict = None, fold: int = 1):
        self.X = X[features] if features is not None else X
        self.fs = features.name
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifiers = classifiers
        self.classifier_params = classifier_params
        self.cross_validation = cv
        self.cv_params = cv_params
        self.predictions = {}
        self.time = {}
        self.fold = fold

        me = modelEvaluation.ModelEvaluation(self.X, self.y)

        match self.cross_validation:
            case 'hold_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.hold_out(**self.cv_params)
            case 'k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.k_fold(**self.cv_params)
            case 'stratified_k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.stratified_k_fold(**self.cv_params)
            case 'leave_one_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.leave_one_out()
            case _:
                raise ValueError('Invalid cross_validation')

        for classifier in self.classifiers:
            match classifier:
                case 'adaboost':
                    self.predictions['adaboost'] = self.adaboost(**self.classifier_params)
                case 'gradient_boosting':
                    self.predictions['gradient boosting'] = self.gradient_boosting(**self.classifier_params)
                case 'random_forest':
                    self.predictions['random forest'] = self.random_forest(**self.classifier_params)
                case 'k_neighbors':
                    self.predictions['k nearest neighbors'] = self.k_nearest_neighbors(**self.classifier_params)
                case 'decision_tree':
                    self.predictions['decision tree'] = self.decision_tree(**self.classifier_params)
                case 'extra_trees':
                    self.predictions['extra trees'] = self.extra_trees(**self.classifier_params)
                case 'svm':
                    self.predictions['svm'] = self.svm(**self.classifier_params)
                case 'xgb':
                    self.predictions['xgb'] = self.xgb(**self.classifier_params)
                case 'all':
                    self.predictions['adaboost'] = self.adaboost(**self.classifier_params)
                    self.predictions['gradient boosting'] = self.gradient_boosting(**self.classifier_params)
                    self.predictions['random forest'] = self.random_forest(**self.classifier_params)
                    self.predictions['k nearest neighbors'] = self.k_nearest_neighbors(**self.classifier_params)
                    self.predictions['decision tree'] = self.decision_tree(**self.classifier_params)
                    self.predictions['extra trees'] = self.extra_trees(**self.classifier_params)
                    self.predictions['svm'] = self.svm(**self.classifier_params)
                    self.predictions['xgb'] = self.xgb(**self.classifier_params)
                case _:
                    raise ValueError('Invalid classifier name')

    def adaboost(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
            adaboostClf = AdaBoostClassifier(
                estimator=kwargs.get('estimator_', None),
                n_estimators=kwargs.get('n_estimators', 50),
                learning_rate=kwargs.get('learning_rate', 1.0),
                algorithm=kwargs.get('algorithm', 'SAMME'),
                random_state=kwargs.get('random_state', 42),
            )
            adaboostClf_f = adaboostClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(adaboostClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['adaboost'] = end_time - start_time

        return predict_proba

    def gradient_boosting(self, **kwargs):

        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
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
                random_state=kwargs.get('random_state', 42),
                max_features=kwargs.get('max_features', None),
                verbose=kwargs.get('verbose', 0),
                max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
                warm_start=kwargs.get('warm_start', False),
                validation_fraction=kwargs.get('validation_fraction', 0.1),
                n_iter_no_change=kwargs.get('n_iter_no_change', None),
                tol=kwargs.get('tol', 1e-4),
                ccp_alpha=kwargs.get('ccp_alpha', 0.0),
            )
            gboostClf_f = gboostClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(gboostClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['gradient boosting'] = end_time - start_time

        return predict_proba

    def random_forest(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
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
                random_state=kwargs.get('random_state', 42),
                verbose=kwargs.get('verbose', 0),
                warm_start=kwargs.get('warm_start', False),
                class_weight=kwargs.get('class_weight', None),
                ccp_alpha=kwargs.get('ccp_alpha', 0.0),
                max_samples=kwargs.get('max_samples', None),
                monotonic_cst=kwargs.get('monotonic_cst', None),
            )
            randomForestClf_f = randomForestClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(randomForestClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['random forest'] = end_time - start_time

        return predict_proba

    def k_nearest_neighbors(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
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
            kneighborsClf_f = kneighborsClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(kneighborsClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['k nearest neighbors'] = end_time - start_time

        return predict_proba

    def decision_tree(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
            dtreeClf = DecisionTreeClassifier(
                criterion=kwargs.get('criterion', 'gini'),
                splitter=kwargs.get('splitter', 'best'),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
                max_features=kwargs.get('max_features', None),
                random_state=kwargs.get('random_state', 42),
                max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
                min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
                class_weight=kwargs.get('class_weight', None),
                ccp_alpha=kwargs.get('ccp_alpha', 0.0),
                monotonic_cst=kwargs.get('monotonic_cst', None),
            )
            dtreeClf_f = dtreeClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(dtreeClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['decision tree'] = end_time - start_time

        return predict_proba

    def extra_trees(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
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
                random_state=kwargs.get('random_state', 42),
                verbose=kwargs.get('verbose', 0),
                warm_start=kwargs.get('warm_start', False),
                class_weight=kwargs.get('class_weight', None),
                ccp_alpha=kwargs.get('ccp_alpha', 0.0),
                max_samples=kwargs.get('max_samples', None),
                monotonic_cst=kwargs.get('monotonic_cst', None),
            )
            extraTreeClf_f = extraTreeClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(extraTreeClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['extra trees'] = end_time - start_time

        return predict_proba

    def svm(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
            svmClf = SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                degree=kwargs.get('degree', 3),
                gamma=kwargs.get('gamma', 'scale'),
                coef0=kwargs.get('coef0', 0.0),
                shrinking=kwargs.get('shrinking', True),
                probability=kwargs.get('probability', False),
                tol=kwargs.get('tol', 1e-3),
                cache_size=kwargs.get('cache_size', 200),
                class_weight=kwargs.get('class_weight', None),
                verbose=kwargs.get('verbose', False),
                max_iter=kwargs.get('max_iter', -1),
                decision_function_shape=kwargs.get('decision_function_shape', 'ovr'),
                break_ties=kwargs.get('break_ties', False),
                random_state=kwargs.get('random_state', 42),
            )
            svmClf_f = svmClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(svmClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['svm'] = end_time - start_time

        return predict_proba

    def xgb(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
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
                random_state=kwargs.get('random_state', 42),
                seed=kwargs.get('seed', None),
                missing=kwargs.get('missing', None),
            )
            xgbClf_f = xgbClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(xgbClf_f.predict(self.X_test[fold]))

        end_time = time.time()
        self.time['xgb'] = end_time - start_time

        return predict_proba
