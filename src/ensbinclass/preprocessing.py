import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class DataPreprocessing:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None

    def load_data(self, path: str, index_col: bool = False, encode: bool = False, sep: str = ','):
        file_extension = os.path.splitext(path)[1]

        if file_extension == '.csv':
            if index_col:
                self.data = pd.read_csv(path, index_col=0, sep=sep)
            else:
                self.data = pd.read_csv(path, sep=sep)
        elif file_extension == '.txt':
            self.data = pd.read_csv(path, sep=sep)
        else:
            raise ValueError('Unsupported file extension, must be .csv or .txt file')

        if encode:
            for col in self.data.columns:
                self.one_hot_encoder(col)
        else:
            print("All columns are numeric")

    def one_hot_encoder(self, col_name: str):
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_data = encoder.fit_transform(self.data[[col_name]])
        encoded_df = pd.DataFrame(encoded_data, columns=[col_name])
        self.data = self.data.drop([col_name], axis=1)
        self.data = pd.concat([self.data, encoded_df], axis=1)

    def set_target(self, col_name: str):
        if col_name not in self.data.columns:
            raise ValueError(f"{col_name} not found in DataFrame columns")

        new_order = [col_name] + [col for col in self.data.columns if col != col_name]
        self.data = self.data[new_order]

        self.y = self.data[col_name]
        self.X = self.data.drop(col_name, axis=1)

        return self.X, self.y

    def normalization(self):
        scaler = MinMaxScaler()
        scaler.fit(self.X)
        self.X = pd.DataFrame(scaler.transform(self.X), index=self.X.index, columns=self.X.columns)
        return self.X

    def remove_collinear_features(self, threshold: float = 0.75):
        col_corr = set()
        corr_matrix = self.X.corr("spearman")
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
                    if colname in self.X.columns:
                        self.X = self.X.drop(colname, axis=1)

    def show_data(self, n_cols: int = 5):
        print(self.data.head(n_cols))
