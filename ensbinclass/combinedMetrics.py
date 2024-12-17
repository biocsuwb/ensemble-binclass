import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from src.performanceMetrics import PerformanceMetrics


class CombinedMetrics:
    def __init__(self, performanceMetrics: List[PerformanceMetrics], classifiers: List[str]):
        self.performanceMetrics = performanceMetrics
        self.classifiers = classifiers

    def plot_classifiers_acc(self):
        for clf in self.classifiers:
            scores_dict = {}
            for metric in self.performanceMetrics:
                scores_dict[metric.fs] = metric.accuracy_score()[1][clf]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=list(scores_dict.values()), palette='hls')
            plt.xticks(ticks=range(len(scores_dict)), labels=list(scores_dict.keys()), rotation=90)
            plt.ylabel('Accuracy score')
            plt.title(f'Box plot of FS accuracy, classifier: {clf}')
            plt.grid(True)
            sns.set_theme()

            plt.show()
