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
    # mse_explanation = "预测值与实际值之间差异的平方的平均值。"
    # rmse_explanation = "均方误差的平方根，用于度量预测值与实际值之间的平均差异。"
    # mape_explanation = "预测值与实际值之间差异的百分比的平均值。"

    # table = f"""
    #     <table>
    #         <tr>
    #             <th>指标</th>
    #             <th>数学解释</th>
    #         </tr>
    #         <tr>
    #             <td>MSE</td>
    #             <td>{mse_explanation}</td>
    #         </tr>
    #         <tr>
    #             <td>RMSE</td>
    #             <td>{rmse_explanation}</td>
    #         </tr>
    #         <tr>
    #             <td>MAPE</td>
    #             <td>{mape_explanation}</td>
    #         </tr>
    #     </table>
    # """
    
    headers = "<tr><th></th>" + "".join(f"<th>{name}</th>" for name in ["mae", "rsme", "mape"]) + "</tr>"
    rows1 = "<tr><td>前30%</td>" + "".join(f"<td>{item:.2f}</td>" for item in data1) + "</tr>"
    rows2 = "<tr><td>后70%</td>" + "".join(f"<td>{item:.2f}</td>" for item in data2) + "</tr>"
    return f"<table>{headers}{rows1}{rows2}</table>"

def plot_result(x, input, gt, y, var, model_choice=None):
    fig = plt.figure(dpi=500)

    if model_choice == "Gaussian Process":
        plt.fill_between(
            x,
            np.ravel(y + 2.5 * np.sqrt(var)),
            np.ravel(y - 2.5 * np.sqrt(var)),
            alpha=0.3,
            color="red",
            label="99% CI",
        )
        y_high = y + 2.5 * np.sqrt(var)
        mid = len(x[x<=230])
        frot_max, frot_argmax = y_high[:mid].max(), y_high[:mid].argmax()
        last_max, last_argmax = y_high[mid:].max(), y_high[mid:].argmax() + mid
        plt.scatter(x[frot_argmax], frot_max, label="max_power1", linewidths=1)
        plt.scatter(x[last_argmax], last_max, label="max_power2", linewidths=1)
    else:
        mid = len(x[x<=230])
        frot_max, frot_argmax = y[:mid].max(), y[:mid].argmax()
        last_max, last_argmax = y[mid:].max(), y[mid:].argmax() + mid
        plt.scatter(x[frot_argmax], frot_max, label="max_power1", linewidths=1)
        plt.scatter(x[last_argmax], last_max, label="max_power2", linewidths=1)
  
    det_normal = "异常样本❌" if (frot_max > 40) or (last_max > 47) else "正常样本✅"
        
    plt.plot(x[x<=230], np.array([40] * len(x[x<=230])), label="threshold1")
    plt.plot(x[x>230], np.array([47] * len(x[x>230])), label="threshold2")
    
    plt.plot(x, gt, label="gt", alpha=0.5)
    plt.plot(x, y, label="pred", alpha=0.5)
    plt.plot(x, input, label="trad", alpha=0.7)
    plt.axvline(x[len(x[x<=230])], color='purple', linestyle='--', label="30% line")

    plt.xlabel("Frequency")
    plt.ylabel("Power")
    
    plt.legend(loc="best", prop={'size': 5})
    plt.xlim(25, 1250)

    return fig, det_normal


def main_function(model_choice, file1, file2):
    file1 = data_process(file1.name)
    file2 = data_process(file2.name)

    if model_choice == "Gaussian Process":
        res = gp_infer(file1, file2)
    elif model_choice == "SVR":
        res = svr_infer(file1, file2)
    elif model_choice == "LightGBM":
        res = lgb_infer(file1, file2)
    
    gt, pred, l = res[2], res[3], len(res[1])
    
    fst_metric = [m(gt[:int(.3*l)], pred[:int(.3*l)]) for m in metrics]
    sec_metric = [m(gt[int(.3*l):], pred[int(.3*l):]) for m in metrics]

    fig, det_normal = plot_result(*res, model_choice=model_choice)

    return fig, det_normal, create_html_table(fst_metric, sec_metric)

iface = gr.Interface(
    fn=main_function,
    inputs=[
        gr.Dropdown(
            choices=["Gaussian Process", "LightGBM"], label="模型选择"
        ),
        gr.File(label="传统测试数据"),
        gr.File(label="新测试数据"),
    ],
    outputs=[gr.Plot(label="可视化图像"), gr.Textbox(label="异常检测结果"), gr.HTML(value="数值可视化区域", show_label=True)],
    examples=[
        ["Gaussian Process", "../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        ["Gaussian Process", "../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        ["Gaussian Process", "../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        ["Gaussian Process", "../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
        ["LightGBM", "../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        ["LightGBM", "../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        ["LightGBM", "../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        ["LightGBM", "../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
        # ["SVR", "../data/13DKB_trad/1H", "../data/13DKB_new/1H"],
        # ["SVR", "../data/13DKB_trad/1V", "../data/13DKB_new/1V"],
        # ["SVR", "../data/13DKB_trad/2H", "../data/13DKB_new/2H"],
        # ["SVR", "../data/13DKB_trad/2V", "../data/13DKB_new/2V"],
    ],
    examples_per_page=8,
    theme=gr.themes.Soft(),
    title=TITLE,
    description=DESCRIPTION,
)
iface.launch(share=False)
