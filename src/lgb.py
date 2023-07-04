import numpy as np
import matplotlib.pyplot as plt
from utils.evaluate import np_mae, np_mape, np_rmse
from sklearn import svm
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from lgb_data import MideaData
import pickle
from sklearnex import patch_sklearn, unpatch_sklearn
import logging
import sys
import lightgbm as lgb
import time
import os
import warnings
from scipy import interpolate
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import pandas as pd
from openpyxl import load_workbook

# test git
def get_dataset(test=1, test_range=[0, 6000], win=7, num_mount=20, scale=1):
    data = MideaData(align=False, scale=scale)
    data.gen_mount_data(win=win, num_mount=num_mount)
    data.plot_data()
    # get_data(self, cls: str = "13DKB", type: str = "1H", ratio: float = 0.25):

    train_new, train_trad, test_new, test_trad = data.get_data(cls='13DKB', test=test, test_range=test_range)
    train_new[:, 0] /= 1e6
    train_trad[:, 0] /= 1e6
    test_new[:, 0] /= 1e6
    test_trad[:, 0] /= 1e6

    train_data = lgb.Dataset(train_new, label=train_trad[:,1])
    return train_data, test_new, test_trad[:, 1]

def mylog(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not os.path.exists(path):
        os.mkdir(path)
    fh = logging.FileHandler(path + "/test.log")
    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("start")
    return logger

# get_dataset()
lgm_paras = {
                "learning_rate": 0.001,
                "num_leaves": 50,
                "n_estimators": 8000,
                "max_depth": 10
}


def main():
    logger = mylog(path='lgb_log')
    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting": "goss",
        "learning_rate": lgm_paras["learning_rate"],
        "num_leaves": lgm_paras["num_leaves"],
        "n_estimators": lgm_paras["n_estimators"],
        "max_depth": lgm_paras["max_depth"],
        "num_threads": 16,
        "device": "gpu",
        "gpu_platform_id": 0,
        "verbose": -1,
    }
    for s in [1.5]:
        win = 15
        num_mount = 30
        scale = s

        logger.info(f'win:{win}, num_mount:{num_mount}, scale:{scale}')
        former_mae, latter_mae = [], []
        former_mape, latter_mape = [], []
        former_rmse, latter_rmse = [], []
        for t in range(1, 14):
        # for t in [2]:
            for test_range in ([0, 6000], [6000, 20000]):
                train_data, test_x, test_y = get_dataset(test=t, test_range=test_range, win=win, num_mount=num_mount, scale=scale)
                model = lgb.train(params, train_data)
                pre = model.predict(test_x)


                mae = np_mae(pre, test_y)
                mape = np_mape(pre, test_y)
                rmse = np_rmse(pre, test_y)

                figure = plt.figure()
                plt.plot(list(range(len(pre))), pre, c='y', label='new', zorder=2)
                plt.plot(list(range(len(pre))), test_y, c='b', label='gt', zorder=1)
                plt.legend()
                plt.title(f'{t}_{test_range}{mae}')
                if test_range == [0, 6000]:
                    part = 'former'
                else:
                    part = 'latter'
                name = f't{t}_' + part + f'_win{win}_num{num_mount}_scale{scale}'
                plt.savefig(f'./lgb_log/{name}.jpg')
                if test_range == [0, 6000]:
                    former_mae.append(mae)
                    former_mape.append(mape)
                    former_rmse.append(rmse)
                else:
                    latter_mae.append(mae)
                    latter_mape.append(mape)
                    latter_rmse.append(rmse)

                logger.info('-'*30 + 'test{}-range{}'.format(t, test_range) + '-' * 30)
                logger.info(params)
                logger.info(f'mae:{mae} mape:{mape} rmse:{rmse}')
                logger.info('-'*30 + '-' + '-' * 30 + '\n')
     # 假设数据存储在变量 data 中，以字典的形式表示
    # data = {
    #     'A_MAE': former_mae,
    #     'A_MAPE': former_mape,
    #     'A_RMSE': former_rmse,
    #     'B_MAE': latter_mae,
    #     'B_MAPE': latter_mape,
    #     'B_RMSE': latter_rmse
    # }
    # print(data)
    # logger.info(data)
    # # 创建 DataFrame
    # df = pd.DataFrame(data)

    # # 设置 Excel 文件名和工作表名
    # excel_file = 'data_results.xlsx'
    # sheet_name = 'Sheet1'
    # # 尝试加载现有的 Excel 文件
    # try:
    #     # 打开现有的 Excel 文件
    #     book = load_workbook(excel_file)
    #     writer = pd.ExcelWriter(excel_file, engine='openpyxl')
    #     writer.book = book
        
    #     # 获取现有工作表
    #     writer.sheets = {ws.title: ws for ws in book.worksheets}
    # # 如果文件不存在，则创建新的 Excel 文件
    # except FileNotFoundError:
    #     writer = pd.ExcelWriter(excel_file, engine='openpyxl')
    # # 将数据追加到现有工作表中或创建新工作表
    # df.to_excel(writer, index=False)
    # # 保存数据到 Excel 文件
    # writer.save()
main()



