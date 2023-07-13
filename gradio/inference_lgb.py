import gradio as gr
from gradio.components import File, Image
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import lightgbm as lgb
from utils import np_mae, np_rmse, np_mape

metrics = [np_mae, np_rmse, np_mape]

model_path = './lgm_params_2.txt'
def inference_lgm(trad_file, new_file):

    trad = np.genfromtxt(trad_file, delimiter=";")[:, :2]
    new = np.genfromtxt(new_file)

    x_new = np.linspace(30000000, 1000000000, 20001)
    xi = np.concatenate([new[:, 0], np.array([1000000000])])
    yi = np.concatenate([new[:, 1], np.array([new[-1, 1]])])
    interp = interpolate.interp1d(xi, yi, kind = "cubic")
    y_new = interp(x_new)
    new_align = np.stack([x_new, y_new], 1)

    new_align[:, 0] /= 1e6
    trad[:, 0] /= 1e6

    model = lgb.Booster(model_file = model_path)

    pre = model.predict(new_align, num_iteration=model.best_iteration)
    
    res = [np.round(f(pre, trad[:, 1]), 3) for f in metrics]

    fig = plt.figure()

    plt.plot(new_align[:, 0], trad[:, 1], label="gt")
    plt.plot(new_align[:, 0], pre, label="pred")

    plt.text(
        0.8,
        0.1,
        "MAE: {}\nRMSE: {}\nMAPE: {}".format(*res),
        ha="left",
        va="bottom",
        transform=plt.gca().transAxes,
    )

    plt.legend(loc="upper left")

    return fig

# inference_lgm('/data/lulongfei/thucode/meidi/Midea_THU/data_1/13DKB_trad/1H', '/data/lulongfei/thucode/meidi/Midea_THU/data_1/13DKB_new/1H')

iface = gr.Interface(
    inference_lgm,
    inputs=[gr.File(), gr.File()],
    outputs=gr.Plot(),
    title="美的-清华Demo(测试版)",
    description="请选取传统数据与新数据文件，点击提交，即可得到预测结果。",
    examples=[
        ["../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        ["../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        ["../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        ["../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
    ],
)


iface.launch(share=True)
