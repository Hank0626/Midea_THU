import gradio as gr
from gradio.components import File, Image
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from utils import np_mae, np_rmse, np_mape

metrics = [np_mae, np_rmse, np_mape]


model_path = "/data3/liupeiyuan/Midea_THU/output/gp_lf_0613/2H/epoch19999/model"



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


def inference(trad_file, new_file):
    trad = np.genfromtxt(trad_file.name, delimiter=";")[:, :2]
    new = np.genfromtxt(new_file.name)

    f = interp1d(new[:, 0], new[:, 1], kind="linear", fill_value="extrapolate")
    
    y_interpolate = f(trad[:, 0])
    
    data = np.c_[trad[:, 0], y_interpolate, trad[:, 1]]
    
    data[:, 0] /= 1e6 
    
    model = tf.saved_model.load(model_path)
    
    mean, _ = model.compiled_predict_f(expand_data(data, 50)[:, :-1])
    
    y = mean.numpy().reshape(-1)
    
    res = [np.round(f(y, data[:, -1]), 3) for f in metrics]
    
    fig = plt.figure()
    
    plt.plot(data[:, 0], data[:, 2], label="gt")
    plt.plot(data[:, 0], y, label="pred")
    
    plt.text(1, 1, "MAE: {}\nRMSE: {}\nMAPE: {}".format(*res))

    plt.legend(loc="upper left")

    return fig


iface = gr.Interface(
    inference, 
    inputs = [gr.File(), gr.File()], 
    outputs=gr.Plot(),
    title="美的-清华Demo",
    description="请选取传统数据与新数据文件，点击提交，即可得到预测结果。"
)


iface.launch(share=True)
