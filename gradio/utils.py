import csv
import numpy as np
import pandas as pd
import math

def np_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def np_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def np_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None)))


def data_process(file):
    if file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".xlsm"):
        data = pd.read_excel(file)
        data = data[data.columns[:2]]
        mask = data.applymap(lambda x: x.isnumeric() if isinstance(x, str) else True)
        data = data[mask.all(axis=1)].values.astype(np.float64)
    elif file.endswith(".txt"):
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)

            reader = csv.reader(f, dialect)

            data = np.array([row for row in reader]).astype(np.float64)
    else:
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)

            reader = csv.reader(f, dialect)

            data = np.array([row for row in reader])[:, :2].astype(np.float64)
            
    min_value = 10
    max_value = 1500

    min_exponent = int(math.ceil(math.log10(min_value / min(data[:, 0]))))
    max_exponent = int(math.floor(math.log10(max_value / max(data[:, 0]))))
    exponent = min(min_exponent, max_exponent)
    data[:, 0] = data[:, 0] * 10**exponent

    return data


# data = data_process("/data3/liupeiyuan/Midea_THU/data/13DKB_trad/1H")
# print(data)


def filter_processing(
    data,
    filter_win = 200,
    peak_thresh = 20,
    peak_space = 300
    ):
    """
    中位数滤波并筛选凸起
    
    参数:
        data (list): 传统设备数据(m, 3)。第一列为频率, 第二列为新数据，第三列为传统测量值
        
        filter_win (int): 中位数滤波窗口大小
        peak_thresh (int): 凸起阈值。
        peak_space (int): 凸起间隔，用于筛选过于稠密的凸起。

    返回:
        filter_data (list): 基线训练数据列表,第一列为x, 第二列为y_new, 第三列为y_trad
        peak_list (list): 突起测试数据列表, 第一列为突起区域与极限差值最大处idx, 第二列为差值delta
    """
    
    # filter
    filter_data = []
    for i in range(data.shape[0]):
        filter_data.append([data[i, 0], \
            np.median(data[max(0,i-filter_win):i+filter_win, 1]), \
            np.median(data[max(0,i-filter_win):i+filter_win, 2])])
    filter_data = np.asarray(filter_data)
    
    
    # peak
    raw_peak_list = []
    for i in range(data.shape[0]):
        if data[i, 1] - filter_data[i, 1] > peak_thresh:
            raw_peak_list.append((i, data[i, 1] - filter_data[i, 1]))
        
    peak_list = []
    for i, delta in raw_peak_list:
        if len(peak_list)==0 or (i - peak_list[-1][0]) > peak_space:            
            peak_list.append([i, delta])
        else:
            if delta > peak_list[-1][1] and (i - peak_list[-1][0]) < peak_space:
                peak_list.pop()
                peak_list.append([i, delta])
    peak_list = np.asarray(peak_list)
    
    return filter_data, peak_list