from utils.mideadata import MideaData
from utils.evaluate import np_mae, np_mape, np_rmse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
import numpy as np


def filter_median(
    cls = "13DKB",
    test_cls = '1H',
    filter_win = 200
    ):
    """
    中位数滤波

    参数:
        cls (str): 分类名称，默认为 "13DKB"。
        test_cls (str): 测试数据集
        filter_win (int): 中位数滤波窗口大小

    返回:
        filter_train_data (list): 训练数据列表, 元素为tuple(name(str), data(array(20001,3)))。data第一列为x, 第二列为y_new, 第三列为y_trad
        filter_test_data (list): 测试数据列表, 元素为tuple(name(str), data(array(20001,3)))。data第一列为x, 第二列为y_new, 第三列为y_trad

    """
    data = MideaData(cls=[cls])
    train_data, test_data = data.get_ori_data(cls=cls, test_cls=test_cls) 
    filter_win = 200

    filter_train_data, filter_abnormal_train_data = [], []
    for name, data in train_data:
        filter_data = []
        for i in range(data.shape[0]):
            filter_data.append([data[i, 0], \
                np.median(data[max(0,i-filter_win):i+filter_win, 1]), \
                np.median(data[max(0,i-filter_win):i+filter_win, 2])])
        filter_data = np.asarray(filter_data)
        filter_train_data.append((name, filter_data, data))
    
    filter_test_data = []
    for name, data in test_data:
        filter_data = []
        for i in range(data.shape[0]):
            filter_data.append([data[i, 0], \
                np.median(data[max(0,i-filter_win):i+filter_win, 1]), \
                np.median(data[max(0,i-filter_win):i+filter_win, 2])])
        filter_data = np.asarray(filter_data)
        filter_test_data.append((name, filter_data, data))       
    
    return filter_train_data, filter_test_data

def filter_peak(
    cls = "13DKB",
    test_cls = '1H',
    filter_win = 200,
    peak_thresh = 20,
    peak_space = 300
    ):
    """
    中位数滤波并筛选凸起
    
    参数:
        cls (str): 分类名称，默认为 "13DKB"。
        test_cls (str): 测试数据集
        filter_win (int): 中位数滤波窗口大小
        peak_thresh (int): 凸起阈值。
        peak_space (int): 凸起间隔，用于筛选过于稠密的凸起。

    返回:
        filter_train_data (list): 基线训练数据列表, 元素为tuple(name(str), filter_data(array(20001,3), data(array(20001,3)))。filter_data第一列为x, 第二列为y_new, 第三列为y_trad
        filter_test_data (list): 基线测试数据列表, 元素为tuple(name(str), filter_data(array(20001,3), data(array(20001,3)))。filter_data第一列为x, 第二列为y_new, 第三列为y_trad
        peak_train_data (list): 突起训练数据列表, 元素为tuple(name(str), peak_list(array(n,2)))。peak_list第一列为突起区域与极限差值最大处idx, 第二列为差值delta
        peak_test_data (list): 突起测试数据列表, 元素为tuple(name(str), peak_list(array(n,2)))。peak_list第一列为突起区域与极限差值最大处idx, 第二列为差值delta
    """
    data = MideaData(cls=[cls])
    train_data, test_data = data.get_ori_data(cls=cls, test_cls=test_cls) 


    # filter
    filter_train_data = []
    for name, data in train_data:
        filter_data = []
        for i in range(data.shape[0]):
            filter_data.append([data[i, 0], \
                np.median(data[max(0,i-filter_win):i+filter_win, 1]), \
                np.median(data[max(0,i-filter_win):i+filter_win, 2])])
        filter_data = np.asarray(filter_data)
        filter_train_data.append((name, filter_data, data))
    
    filter_test_data = []
    for name, data in test_data:
        filter_data = []
        for i in range(data.shape[0]):
            filter_data.append([data[i, 0], \
                np.median(data[max(0,i-filter_win):i+filter_win, 1]), \
                np.median(data[max(0,i-filter_win):i+filter_win, 2])])
        filter_data = np.asarray(filter_data)
        filter_test_data.append((name, filter_data, data))       
    
    
    # peak
    peak_train_list= []   
    for (name, filter_data, data) in filter_train_data:
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
        peak_train_list.append((name, peak_list))
        
    peak_test_list= []   
    for (name, filter_data, data) in filter_test_data:
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
        peak_test_list.append((name, peak_list))   
        
        
    return filter_train_data, filter_test_data, peak_train_list, peak_test_list


def filter_processing(
    trad, 
    new,
    filter_win = 200,
    peak_thresh = 20,
    peak_space = 300
    ):
    """
    中位数滤波并筛选凸起
    
    参数:
        trad (list): 传统设备数据(m, 2)。第一列为频率, 第二列为测量值
        new (list): 新设备数据(n, 2)。第一列为频率, 第二列为测量值
        filter_win (int): 中位数滤波窗口大小
        peak_thresh (int): 凸起阈值。
        peak_space (int): 凸起间隔，用于筛选过于稠密的凸起。

    返回:
        filter_data (list): 基线训练数据列表,第一列为x, 第二列为y_new, 第三列为y_trad
        peak_list (list): 突起测试数据列表, 第一列为突起区域与极限差值最大处idx, 第二列为差值delta
    """
    
    new = new[~np.isnan(new).any(axis=1), :]
    new = new[np.isfinite(new).all(axis=1), :]

    f = interp1d(new[:, 0], new[:, 1], kind="linear", fill_value="extrapolate")

    y_interpolate = f(trad[:, 0])

    data = np.c_[trad[:, 0], y_interpolate, trad[:, 1]]

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
