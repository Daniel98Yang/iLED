from typing import Tuple

import numpy as np
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TSCDataset:
    X: np.ndarray
    y: np.ndarray

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def transform_ts_data(X, scaler, scale_separately=False, fit=True):
    if scale_separately:
        if fit:
            X = scaler.fit_transform(X.transpose(0, 2, 1).reshape(-1, X.shape[1])).reshape(-1, X.shape[2], X.shape[1]).transpose(0, 2, 1)
        else:
            X = scaler.transform(X.transpose(0, 2, 1).reshape(-1, X.shape[1])).reshape(-1, X.shape[2], X.shape[1]).transpose(0, 2, 1)
    else:
        if fit:
            X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        else:
            X = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
    return X


def ds_load(datasets_path, ds_name, train_size=None, val_size=None, scaler=None, scale_separately=False) -> Tuple[TSCDataset, TSCDataset]:
    train_file = datasets_path / ds_name / f"{ds_name}_TRAIN.arff"
    test_file = datasets_path / ds_name / f"{ds_name}_TEST.arff"
    label_encoder = LabelEncoder()

    def arff_to_numpy(file_path):
        data, _ = loadarff(file_path)
        X, y = [], []
        for row in data:
            try:
                x, label = row
            except ValueError:
                # univariate data
                row = row.tolist()
                x = np.array(row[:-1]).reshape(1, -1)
                label = row[-1]
            x = np.array(x.tolist(), dtype="float32")
            X.append(x)
            y.append(label)
        return np.nan_to_num(np.stack(X, axis=0), 0), np.stack(y, axis=0)

    if train_size is None:
        trainX, trainy = arff_to_numpy(train_file)
        testX, testy = arff_to_numpy(test_file)
        trainy = label_encoder.fit_transform(trainy)
        testy = label_encoder.transform(testy)
    else:
        X1, y1 = arff_to_numpy(train_file)
        X2, y2 = arff_to_numpy(test_file)
        X = np.concatenate([X1, X2])
        y = np.concatenate([y1, y2])
        y = label_encoder.fit_transform(y)
        trainX, testX, trainy, testy = train_test_split(X, y, train_size=train_size)

    if val_size:
        trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=val_size)

    if scaler:
        trainX = transform_ts_data(trainX, scaler, scale_separately)
        if val_size:
            valX = transform_ts_data(valX, scaler, scale_separately, fit=False)
        testX = transform_ts_data(testX, scaler, scale_separately, fit=False)

    if val_size:
        return TSCDataset(trainX, trainy), TSCDataset(valX, valy), TSCDataset(testX, testy)

    return TSCDataset(trainX, trainy), TSCDataset(testX, testy)
