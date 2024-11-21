import time
import pandas as pd
import src.modelEvaluation as modelEvaluation
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Ensemble:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None, features: pd.Series = None, ensemble: list = None,
                 classifiers: list = None, cross_validation: str = 'hold_out', fold: int = 1, **kwargs):
        self.X = X[features] if features is not None else X
        self.fs = features.name
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ensemble = ensemble
        self.cross_validation = cross_validation
        self.classifiers = classifiers
        self.model_classifiers = []
        self.predictions = {}
        self.fold = fold
        self.time = {}

        me = modelEvaluation.ModelEvaluation(self.X, self.y)

        match self.cross_validation:
            case 'hold_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.hold_out(0.3)
            case 'k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.k_fold(self.fold)
            case 'stratified_k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.stratified_k_fold(self.fold)
            case 'leave_one_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.leave_one_out()
            case _:
                raise ValueError('Invalid cross_validation')

        for classifier in self.classifiers:
            match classifier:
                case 'adaboost':
                    self.model_classifiers.append(('adaboost', AdaBoostClassifier(algorithm='SAMME')))
                case 'gradient_boosting':
                    self.model_classifiers.append(('gradient boosting', GradientBoostingClassifier()))
                case 'random_forest':
                    self.model_classifiers.append(('random forest', RandomForestClassifier()))
                case 'k_neighbors':
                    self.model_classifiers.append(('k_nearest neighbors', KNeighborsClassifier()))
                case 'decision_tree':
                    self.model_classifiers.append(('decision tree', DecisionTreeClassifier()))
                case 'extra_trees':
                    self.model_classifiers.append(('extra trees', ExtraTreesClassifier()))
                case 'svm':
                    self.model_classifiers.append(('svm', SVC()))
                case 'xgb':
                    self.model_classifiers.append(('xgb', XGBClassifier()))
                case 'all':
                    self.model_classifiers = [('adaboost', AdaBoostClassifier(algorithm='SAMME')),
                                              ('gradient_boosting', GradientBoostingClassifier()),
                                              ('random_forest', RandomForestClassifier()),
                                              ('k_neighbors', KNeighborsClassifier()),
                                              ('decision_tree', DecisionTreeClassifier()),
                                              ('extra_trees', ExtraTreesClassifier()),
                                              ('svm', SVC(kernel='linear')),
                                              ('xgb', XGBClassifier())]
                case _:
                    raise ValueError('Invalid classifier name')

        for ens in self.ensemble:
            match ens:
                case 'voting':
                    self.predictions['voting'] = self.voting(**kwargs)
                case 'bagging':
                    self.predictions['bagging'] = self.bagging()
                case 'stacking':
                    self.predictions['stacking'] = self.stacking()
                case 'all':
                    self.predictions['voting'] = self.voting(**kwargs)
                    self.predictions['bagging'] = self.bagging()
                    self.predictions['stacking'] = self.stacking()

    def voting(self, **kwargs):
        start_time = time.time()

        predict_proba = []
        voting = kwargs.get('voting', 'hard')
        if voting not in ('soft', 'hard'):
            raise Exception('Voting should be soft or hard')
        for fold in range(self.fold):
            Voting = VotingClassifier(estimators=self.model_classifiers, voting=voting)
            Voting.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(Voting.predict(self.X_test[fold]))

        end_time = time.time()

        self.time['voting'] = end_time - start_time

        return predict_proba

    def bagging(self):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
            bagging = BaggingClassifier(estimator=self.model_classifiers[0][1])
            bagging.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(bagging.predict(self.X_test[fold]))

        end_time = time.time()

        self.time['bagging'] = end_time - start_time

        return predict_proba

    def stacking(self):
        start_time = time.time()

        predict_proba = []
        for fold in range(self.fold):
            stacking = StackingClassifier(estimators=self.model_classifiers)
            stacking.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(stacking.predict(self.X_test[fold]))

        end_time = time.time()

        self.time['stacking'] = end_time - start_time

        return predict_proba
