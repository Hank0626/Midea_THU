import numpy as np


def np_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def np_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def np_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None)))
