from scipy.interpolate import interp1d
import numpy as np
import gradio as gr
import tensorflow as tf
from utils import np_mae, np_rmse, np_mape

metrics = [np_mae, np_rmse, np_mape]

model_path = "../model/lpy/model"

def expand_data(data, n):
    expanded_data = []
    for i in range(len(data)):
        row_data = []

        for j in range(i - n, i + n + 1):
            if j < 0:
                row_data.append(data[0, 0])
                continue
            if j >= len(data):
                row_data.append(data[len(data) - 1, 0])
                continue
            row_data.append(data[j, 0])

        for j in range(i - n, i + n + 1):
            if j < 0:
                row_data.append(data[0, 1])
                continue
            if j >= len(data):
                row_data.append(data[len(data) - 1, 1])
                continue
            row_data.append(data[j, 1])

        row_data.append(data[i, 2])
        expanded_data.append(row_data)
    return np.array(expanded_data)


def gp_infer(trad, new):
    new = new[~np.isnan(new).any(axis=1), :]
    new = new[np.isfinite(new).all(axis=1), :]

    f = interp1d(new[:, 0], new[:, 1], kind="linear", fill_value="extrapolate")

    y_interpolate = f(trad[:, 0])

    data = np.c_[trad[:, 0], y_interpolate, trad[:, 1]]

    model = tf.saved_model.load(model_path)

    mean, var = model.compiled_predict_f(expand_data(data, 50)[:, :-1])

    y = mean.numpy().reshape(-1)

    return data[:, 0], y_interpolate, data[:, 2], y, var.numpy().reshape(-1)
