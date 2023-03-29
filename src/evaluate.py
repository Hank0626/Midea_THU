import numpy as np


def np_rmse(y_true, y_pred):
    # flatted mean, the same as attconv paper
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def np_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# np直接计算版MAPE
def np_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None)))
