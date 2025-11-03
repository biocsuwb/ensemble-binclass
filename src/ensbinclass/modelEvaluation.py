import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut


class ModelEvaluation:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def hold_out(self, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=kwargs.get('shuffle', True),
                                                            stratify=kwargs.get('stratify', None),
                                                            random_state=kwargs.get('random_state', None),
                                                            )

        return X_train, X_test, y_train, y_test

    def stratified_k_fold(self, **kwargs):
        skf = StratifiedKFold(
            n_splits=kwargs.get('n_splits', 5),
            shuffle=kwargs.get('shuffle', True),
            random_state=kwargs.get('random_state', None),
        )

        train_index_by_fold, test_index_by_fold = [], []

        for i, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
            train_index_by_fold.append(train_index)
            test_index_by_fold.append(test_index)

        return train_index_by_fold, test_index_by_fold
