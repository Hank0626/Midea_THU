import gradio as gr
import numpy as np

import matplotlib.pyplot as plt
from config import *
from inference_svr import svr_infer
from inference_gp import gp_infer
from inference_lgb import lgb_infer

from utils import np_mae, np_rmse, np_mape, data_process

metrics = [np_mae, np_rmse, np_mape]


def create_html_table(data1, data2):
    mse_explanation = "预测值与实际值之间差异的平方的平均值。"
    rmse_explanation = "均方误差的平方根，用于度量预测值与实际值之间的平均差异。"
    mape_explanation = "预测值与实际值之间差异的百分比的平均值。"

    table = f"""
        <table>
            <tr>
                <th>指标</th>
                <th>数学解释</th>
            </tr>
            <tr>
                <td>MSE</td>
                <td>{mse_explanation}</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{rmse_explanation}</td>
            </tr>
            <tr>
                <td>MAPE</td>
                <td>{mape_explanation}</td>
            </tr>
        </table>
    """
    
    headers = "<tr><th></th>" + "".join(f"<th>{name}</th>" for name in ["mae", "rsme", "mape"]) + "</tr>"
    rows1 = "<tr><td>前30%</td>" + "".join(f"<td>{item}</td>" for item in data1) + "</tr>"
    rows2 = "<tr><td>后70%</td>" + "".join(f"<td>{item}</td>" for item in data2) + "</tr>"
    return f"{table}<table>{headers}{rows1}{rows2}</table>"

def plot_result(x, gt, y, var, model_choice=None):
    fig = plt.figure()

    if model_choice == "Gaussian Process":
        plt.fill_between(
            x,
            np.ravel(y + 2 * np.sqrt(var)),
            np.ravel(y - 2 * np.sqrt(var)),
            alpha=0.3,
            color="C0",
            label="95% Confidence Interval",
        )

    plt.plot(x, gt, label="gt", alpha=0.8)
    plt.plot(x, y, label="pred", alpha=0.8)
    plt.axvline(x[int(.3*len(x))], color='red', linestyle='--', label="30% line")

    plt.xlabel("Frequency")
    plt.ylabel("Power")
    
    plt.legend(loc="lower right")

    return fig


def main_function(model_choice, file1, file2):
    file1 = data_process(file1.name)
    file2 = data_process(file2.name)

    if model_choice == "Gaussian Process":
        res = gp_infer(file1, file2)
    elif model_choice == "SVR":
        res = svr_infer(file1, file2)
    elif model_choice == "LightGBM":
        res = lgb_infer(file1, file2)
    
    gt, pred, l = res[1], res[2], len(res[1])
    
    fst_metric = [m(gt[:int(.3*l)], pred[:int(.3*l)]) for m in metrics]
    sec_metric = [m(gt[int(.3*l):], pred[int(.3*l):]) for m in metrics]

    return plot_result(*res, model_choice=model_choice), create_html_table(fst_metric, sec_metric)

iface = gr.Interface(
    fn=main_function,
    inputs=[
        gr.Dropdown(
            choices=["Gaussian Process", "SVR", "LightGBM"], label="模型选择"
        ),
        gr.File(label="传统测试数据"),
        gr.File(label="新测试数据"),
    ],
    outputs=[gr.Plot(label="可视化图像"), gr.HTML(value="数值可视化区域", show_label=True)],
    examples=[
        ["Gaussian Process", "../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        ["Gaussian Process", "../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        ["Gaussian Process", "../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        ["Gaussian Process", "../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
        ["Light GBM", "../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        ["Light GBM", "../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        ["Light GBM", "../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        ["Light GBM", "../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
        ["SVR", "../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        ["SVR", "../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        ["SVR", "../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        ["SVR", "../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
    ],
    examples_per_page=12,
    theme=gr.themes.Soft(),
    title=TITLE,
    description=DESCRIPTION,
)
iface.launch()
