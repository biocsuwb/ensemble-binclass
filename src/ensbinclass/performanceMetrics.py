import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    matthews_corrcoef, precision_score, f1_score, auc, RocCurveDisplay, \
    recall_score


class PerformanceMetrics:
    def __init__(self, classifier):
        self.results = classifier.results

    def confusion_matrix(self):
        pass

    def accuracy(self):
        acc_lst = []

        for y_pred_prob, y_true in zip(self.results['prediction'], self.results['y_true']):
            y_pred = np.argmax(y_pred_prob[0], axis=1)
            acc_lst.append(accuracy_score(y_true, y_pred))

        self.results['acc'] = acc_lst
        return acc_lst

    def roc_auc(self):
        roc_auc_lst = []

        for y_pred_prob, y_true in zip(self.results['prediction'], self.results['y_true']):
            y_pred = np.argmax(y_pred_prob[0], axis=1)
            roc_auc_lst.append(roc_auc_score(y_true, y_pred))

        self.results['roc_auc'] = roc_auc_lst
        return roc_auc_lst

    def f1_score(self):
        f1_score_lst = []

        for y_pred_prob, y_true in zip(self.results['prediction'], self.results['y_true']):
            y_pred = np.argmax(y_pred_prob[0], axis=1)
            f1_score_lst.append(f1_score(y_true, y_pred))

        self.results['f1_score'] = f1_score_lst
        return f1_score_lst

    def matthews_corrcoef(self):
        mcc_lst = []

        for y_pred_prob, y_true in zip(self.results['prediction'], self.results['y_true']):
            y_pred = np.argmax(y_pred_prob[0], axis=1)
            mcc_lst.append(matthews_corrcoef(y_true, y_pred))

        self.results['mcc'] = mcc_lst
        return mcc_lst

    def precision(self):
        precision_lst = []

        for y_pred_prob, y_true in zip(self.results['prediction'], self.results['y_true']):
            y_pred = np.argmax(y_pred_prob[0], axis=1)
            precision_lst.append(precision_score(y_true, y_pred))

        self.results['precision'] = precision_lst
        return precision_lst

    def recall(self):
        recall_lst = []

        for y_pred_prob, y_true in zip(self.results['prediction'], self.results['y_true']):
            y_pred = np.argmax(y_pred_prob[0], axis=1)
            recall_lst.append(recall_score(y_true, y_pred))

        self.results['recall'] = recall_lst
        return recall_lst

    def get_metrics(self):
        pass

    @staticmethod
    def std(x):
        values = [v[0] for v in x.values()]
        return np.std(values)

    def plot_acc(self):
        pass

    def plot_roc_auc(self):
        for y_pred_prob, y_true, classifier, iter, feature_selection, fold \
                in zip(
                    self.results['prediction'],
                    self.results['y_true'],
                    self.results['classifier'],
                    self.results['iter'],
                    self.results['feature_selection'],
                    self.results['fold'],
                ):
            y_pred_classes = np.argmax(y_pred_prob[0], axis=1)

            fpr, tpr, thresholds = roc_curve(
                y_true=y_true,
                y_score=y_pred_classes
            )
            roc_auc = auc(fpr, tpr)
            disp = RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                estimator_name=
                f'estimator {classifier},'
                f' iter {iter},'
                f' feature_selection {feature_selection},'
                f' fold {fold}',
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
