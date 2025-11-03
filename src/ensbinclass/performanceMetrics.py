import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve, \
    matthews_corrcoef, precision_score, f1_score, auc, RocCurveDisplay
from collections import defaultdict


class PerformanceMetrics:
    def __init__(self, classifier):
        self.y_true = classifier.y_true
        self.y_pred = classifier.predictions
        self.classifiers = classifier.classifiers
        self.features = classifier.features

    def _get_y_pred(self):
        pass

    def confusion_matrix(self):
        pass

    def accuracy_score(self):
        acc_score = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            model_suffix = model_name
            y_true_current = self.y_true[model_suffix]

            accuracy = accuracy_score(y_true_current, y_pred_classes)
            acc_score[model_name] = accuracy

        return acc_score

    def roc_auc(self):
        roc_auc = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            y_true_current = self.y_true[model_name]

            accuracy = roc_auc_score(y_true_current, y_pred_classes)
            roc_auc[model_name] = accuracy

        return roc_auc

    def f1_score(self):
        pass

    def matthews_corrcoef(self):
        pass

    def precision_score(self):
        pass

    @staticmethod
    def std(x):
        values = [v[0] for v in x.values()]
        return np.std(values)

    def plot_acc(self):
        pass

    def plot_roc_auc(self):
        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)

            fpr, tpr, thresholds = roc_curve(
                y_true=self.y_true[model_name],
                y_score=y_pred_classes
            )
            roc_auc = auc(fpr, tpr)
            disp = RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                estimator_name=f'estimator {model_name}'
            )
            disp.plot()

    def plot_f1_score(self):
        pass

    def plot_mcc(self):
        pass

    def plot_precision_score(self):
        pass

    def plot_classifier_time(self):
        pass

    def all_metrics(self):
        return [
            self.accuracy_score()[0],
            self.roc_auc()[0],
            self.f1_score()[0],
            self.matthews_corrcoef()[0],
            self.precision_score()[0],
        ]

    def plot_all(self):
        self.plot_acc(),
        self.plot_roc_auc(),
        self.plot_f1_score(),
        self.plot_mcc(),
        self.plot_precision_score(),
        self.plot_classifier_time(),
