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