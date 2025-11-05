import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    matthews_corrcoef, precision_score, f1_score, auc, RocCurveDisplay, \
    recall_score


class PerformanceMetrics:
    def __init__(self, classifier):
        self.y_true = classifier.y_true
        self.y_pred = classifier.predictions
        self.classifiers = classifier.classifiers
        self.features = classifier.features

    def confusion_matrix(self):
        pass

    def accuracy(self):
        acc_score_dict = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            model_suffix = model_name
            y_true_current = self.y_true[model_suffix]

            accuracy = accuracy_score(y_true_current, y_pred_classes)
            acc_score_dict[model_name] = accuracy

        return acc_score_dict

    def roc_auc(self):
        roc_auc_dict = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            y_true_current = self.y_true[model_name]

            accuracy = roc_auc_score(y_true_current, y_pred_classes)
            roc_auc_dict[model_name] = accuracy

        return roc_auc_dict

    def f1_score(self):
        f1_score_dict = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            model_suffix = model_name
            y_true_current = self.y_true[model_suffix]

            accuracy = f1_score(y_true_current, y_pred_classes)
            f1_score_dict[model_name] = accuracy

        return f1_score_dict

    def matthews_corrcoef(self):
        mcc_dict = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            model_suffix = model_name
            y_true_current = self.y_true[model_suffix]

            accuracy = matthews_corrcoef(y_true_current, y_pred_classes)
            mcc_dict[model_name] = accuracy

        return mcc_dict

    def precision(self):
        precision_score_dict = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            model_suffix = model_name
            y_true_current = self.y_true[model_suffix]

            accuracy = precision_score(y_true_current, y_pred_classes)
            precision_score_dict[model_name] = accuracy

        return precision_score_dict

    def recall(self):
        recall_score_dict = {}

        for model_name, pred_probs_list in self.y_pred.items():
            pred_probs = pred_probs_list[0]
            y_pred_classes = np.argmax(pred_probs, axis=1)
            model_suffix = model_name
            y_true_current = self.y_true[model_suffix]

            accuracy = recall_score(y_true_current, y_pred_classes)
            recall_score_dict[model_name] = accuracy

        return recall_score_dict

    def get_metrics(self):
        acc_dict = self.accuracy()
        precision_dict = self.precision()
        recall_dict = self.recall()

        rows = []
        all_keys = set(acc_dict) | set(precision_dict) | set(recall_dict)

        for model_key in sorted(all_keys):
            parts = model_key.split('-', 3)
            classifier = parts[0].lower()
            repeat = int(parts[1])
            feature_name = parts[2]
            fold = int(parts[3])
            selected_features = self.features[repeat].tolist()

            rows.append({
                "classifier": classifier,
                "feature selection": feature_name,
                "repeat": repeat,
                "fold": fold,
                "accuracy": acc_dict.get(model_key),
                "precision": precision_dict.get(model_key),
                "recall": recall_dict.get(model_key),
                "selected_features": selected_features,
            })

        return pd.DataFrame(rows, columns=[
            "classifier", "feature selection",  "repeat", "fold",
            "accuracy", "precision", "recall", "selected_features"
        ])

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
            self.accuracy()[0],
            self.roc_auc()[0],
            self.f1_score()[0],
            self.matthews_corrcoef()[0],
            self.precision()[0],
            self.recall()[0],
        ]

    def plot_all(self):
        self.plot_acc(),
        self.plot_roc_auc(),
        self.plot_f1_score(),
        self.plot_mcc(),
        self.plot_precision_score(),
        self.plot_classifier_time(),
