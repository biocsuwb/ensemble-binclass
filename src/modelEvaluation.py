import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut


class ModelEvaluation:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def _split_data(self, train_index, test_index):
        X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
        y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

        return X_train, X_test, y_train, y_test

    def _split_data_and_append(self, split_indices):
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        for train_index, test_index in split_indices:
            X_train, X_test, y_train, y_test = self._split_data(train_index, test_index)

            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

        return X_train_list, X_test_list, y_train_list, y_test_list

    def hold_out(self, test_size: float):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test

    def k_fold(self, n_splits: int):
        kf = KFold(n_splits=n_splits)
        split_indices = list(kf.split(self.X))

        X_train_list, X_test_list, y_train_list, y_test_list = self._split_data_and_append(split_indices)

        return X_train_list, X_test_list, y_train_list, y_test_list

    def stratified_k_fold(self, n_splits: int):
        skf = StratifiedKFold(n_splits=n_splits)
        split_indices = list(skf.split(self.X, self.y))

        X_train_list, X_test_list, y_train_list, y_test_list = self._split_data_and_append(split_indices)

        return X_train_list, X_test_list, y_train_list, y_test_list

    def leave_one_out(self):
        loo = LeaveOneOut()
        split_indices = list(loo.split(self.X))

        X_train_list, X_test_list, y_train_list, y_test_list = self._split_data_and_append(split_indices)

        return X_train_list, X_test_list, y_train_list, y_test_list
