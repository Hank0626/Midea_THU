from scipy.interpolate import interp1d
import numpy as np
import pickle
import os

model_path = "../gradio_model/svr"

def svr_infer(trad, new):
    new = new[~np.isnan(new).any(axis=1), :]
    new = new[np.isfinite(new).all(axis=1), :]

    f = interp1d(new[:, 0], new[:, 1], kind="linear", fill_value="extrapolate")

    y_interpolate = f(trad[:, 0])

    data = np.c_[trad[:, 0], y_interpolate, trad[:, 1]]

    data[:, 0] /= 1e2

    mid = int(len(data) * (4124/20000))

    data1 = data[:mid, :]
    data2 = data[mid:, :]
    
    with open(os.path.join(model_path, "best1.pkl"), "rb") as f:
        model1 = pickle.load(f)

    with open(os.path.join(model_path, "best2.pkl"), "rb") as f:
        model2 = pickle.load(f)

    pre1 = model1.predict(data1[:, :2])
    pre2 = model2.predict(data2[:, :2])

    y = np.concatenate((pre1, pre2), axis=0)

    return data[:, 0], y_interpolate, data[:, 2], y, 0
