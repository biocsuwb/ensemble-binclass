import time
import pandas as pd
import modelEvaluation
from performanceMetrics import PerformanceMetrics
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Ensemble:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None, features: list = None,
                 classifiers: list = None, classifier_params: list = None, cv: str = 'hold_out',
                 cv_params: dict = None, ensemble: list = None, ensemble_params: list = None, repetitions: int = 1):
        self.X = X
        self.y = y
        self.fs = None
        self.features = features
        self.classifiers = classifiers
        self.classifier_params = classifier_params
        if self.classifier_params is None:
            self.classifier_params = [{classifier: {}} for classifier in self.classifiers]
        self.model_classifiers = {}
        self.cross_validation = cv
        self.cv_params = cv_params if cv_params is not None else {}
        self.ensemble = ensemble
        self.ensemble_params = ensemble_params
        if self.ensemble_params is None:
            self.ensemble_params = [{ensemble: {}} for ensemble in self.ensemble]

        self.predictions = {}
        self.time = {}
        self.n_splits = self.cv_params.get('n_splits', 5)
        self.repetitions = repetitions

        for i in range(self.repetitions):
            for feature_set in self.features:
                X_ = self.X[feature_set]
                self.fs = feature_set.name

                me = modelEvaluation.ModelEvaluation(X_, self.y)

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

                for classifier, params in zip(self.classifiers, self.classifier_params):
                    match classifier:
                        case 'adaboost':
                            self.model_classifiers['adaboost'] = AdaBoostClassifier(**params['adaboost'])
                        case 'gradient_boosting':
                            self.model_classifiers['gradient_boosting'] = GradientBoostingClassifier(**params['gradient_boosting'])
                        case 'random_forest':
                            self.model_classifiers['random_forest'] = RandomForestClassifier(**params['random_forest'])
                        case 'k_neighbors':
                            self.model_classifiers['k_neighbors'] = KNeighborsClassifier(**params['k_neighbors'])
                        case 'decision_tree':
                            self.model_classifiers['decision_tree'] = DecisionTreeClassifier(**params['decision_tree'])
                        case 'extra_trees':
                            self.model_classifiers['extra_trees'] = ExtraTreesClassifier(**params['extra_trees'])
                        case 'svm':
                            params['svm']['probability'] = True
                            self.model_classifiers['svm'] = SVC(**params['svm'])
                        case 'xgb':
                            self.model_classifiers['xgb'] = XGBClassifier(**params['xgb'])
                        case 'all':
                            self.model_classifiers['adaboost'] = AdaBoostClassifier(**params['adaboost'])
                            self.model_classifiers['gradient_boosting'] = GradientBoostingClassifier(**params['gradient_boosting'])
                            self.model_classifiers['random_forest'] = RandomForestClassifier(**params['random_forest'])
                            self.model_classifiers['k_neighbors'] = KNeighborsClassifier(**params['k_neighbors'])
                            self.model_classifiers['decision_tree'] = DecisionTreeClassifier(**params['decision_tree'])
                            self.model_classifiers['extra_trees'] = ExtraTreesClassifier(**params['extra_trees'])
                            self.model_classifiers['svm'] = SVC(**params['svm'])
                            self.model_classifiers['xgb'] = XGBClassifier(**params['xgb'])
                        case _:
                            raise ValueError('Invalid classifier name')

                for ensemble, params in zip(self.ensemble, self.ensemble_params):
                    match ensemble:
                        case 'voting':
                            self.predictions[f'{feature_set.name}_VOTING_{i}'] = self.voting(**params['voting'])
                        case 'bagging':
                            self.predictions[f'{feature_set.name}_BAGGING_{i}'] = self.bagging(**params['bagging'])
                        case 'stacking':
                            self.predictions[f'{feature_set.name}_STACKING_{i}'] = self.stacking(**params['stacking'])
                        case 'all':
                            self.predictions[f'{feature_set.name}_VOTING_{i}'] = self.voting(**params['voting'])
                            self.predictions[f'{feature_set.name}_BAGGING_{i}'] = self.bagging(**params['bagging'])
                            self.predictions[f'{feature_set.name}_STACKING_{i}'] = self.stacking(**params['stacking'])

    def voting(self, **kwargs):
        estimators = [(name, clf) for name, clf in self.model_classifiers.items()]

        start_time = time.time()

        predict_proba = []
        for fold in range(self.n_splits):
            Voting = VotingClassifier(
                estimators=estimators,
                voting=kwargs.get('voting', 'hard'),
                weights=kwargs.get('weights', None),
                n_jobs=kwargs.get('n_jobs', None),
                flatten_transform=kwargs.get('flatten_transform', True),
                verbose=kwargs.get('verbose', False),
            )
            Voting.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(Voting.predict(self.X_test[fold]))

        end_time = time.time()

        self.time['voting'] = end_time - start_time

        return predict_proba

    def bagging(self, **kwargs):
        estimator = self.model_classifiers.get(kwargs.get('estimator_name'), None)
    
        start_time = time.time()

        predict_proba = []
        for fold in range(self.n_splits):
            bagging = BaggingClassifier(
                estimator=estimator,
                n_estimators=kwargs.get('n_estimators', 10),
                max_samples=kwargs.get('max_samples', 1.0),
                max_features=kwargs.get('max_features', 1.0),
                bootstrap=kwargs.get('bootstrap', True),
                bootstrap_features=kwargs.get('bootstrap_features', False),
                oob_score=kwargs.get('oob_score', False),
                warm_start=kwargs.get('warm_start', False),
                n_jobs=kwargs.get('n_jobs', None),
                random_state=kwargs.get('random_state', None),
                verbose=kwargs.get('verbose', 0),
            )
            bagging.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(bagging.predict(self.X_test[fold]))

        end_time = time.time()

        self.time['bagging'] = end_time - start_time

        return predict_proba

    def stacking(self, **kwargs):
        estimators = [(name, clf) for name, clf in self.model_classifiers.items()]

        start_time = time.time()

        predict_proba = []
        for fold in range(self.n_splits):
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=kwargs.get('final_estimator', None),
                cv=kwargs.get('cv', None),
                stack_method=kwargs.get('stack_method', 'auto'),
                n_jobs=kwargs.get('n_jobs', None),
                passthrough=kwargs.get('passthrough', False),
                verbose=kwargs.get('verbose', 0),
            )
            stacking.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(stacking.predict(self.X_test[fold]))

        end_time = time.time()

        self.time['stacking'] = end_time - start_time

        return predict_proba

    def f1_score(self):
        pm = PerformanceMetrics(self)
        return pm.f1_score()

    def accuracy_score(self):
        pm = PerformanceMetrics(self)
        return pm.accuracy_score()

    def roc_auc(self):
        pm = PerformanceMetrics(self)
        return pm.roc_auc()

    def matthews_corrcoef(self):
        pm = PerformanceMetrics(self)
        return pm.matthews_corrcoef()

    def confusion_matrix(self):
        pm = PerformanceMetrics(self)
        return pm.confusion_matrix()

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
