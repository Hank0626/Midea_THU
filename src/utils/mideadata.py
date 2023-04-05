import os
import os.path as osp
import numpy as np
import time


# 装饰器，用于输出数据统计信息
def data_statistics(f):
    def wrapper(*args, **kwargs):
        new, train, test = f(*args, **kwargs)
        print(kwargs)
        print(f"<========= Loading {kwargs['cls']}:{kwargs['type']} =========>")
        print(f"new shape: {new.shape=}")  # 输出新设备数据的形状
        print(f"train shape: {train.shape=}")  # 输出传统设备训练数据的形状
        print(f"test shape: {test.shape=}")  # 输出传统设备测试数据的形状
        print(f"<========= Data Loaded =========>")
        time.sleep(3)
        return new, train, test

    return wrapper


class MideaData(object):
    """
    MideaData类用于处理和组织数据。该类包括两个主要的数据字典: trad_data和new_data, 分别用于存储传统数据和新数据。

    数据字典结构如下：

    trad_data: {
        '13DKB': {
            '1H{xx}': <numpy.ndarray 数据>,
            '2H{xx}': <numpy.ndarray 数据>,
            ...
        }
    }

    new_data: {
        '13DKB': {
            '1H': <numpy.ndarray 数据>,
            '2H': <numpy.ndarray 数据>,
            ...
        }
    }

    其中，'13DKB'是类别(cls)，可以根据实际情况进行扩展。数据字典的键是从数据文件夹(如：../data/13DKB_trad和../data/13DKB_new)中读取的文件名。
    对于每个类别，文件名的格式为"{type}{xx}"（传统数据）和"{type}"（新数据），如：'1H4M' 和 '1H'。
    文件名中的{type}表示数据的类型，例如：'1H'、'2H'等。

    在初始化MideaData对象时, 它会自动读取数据文件夹中的所有数据, 并将其存储为numpy数组。
    """

    def __init__(self) -> None:
        self.data_path = "../data/"
        self.cls = ["13DKB"]
        self.trad_data = dict()  # 传统设备数据
        self.new_data = dict()  # 新设备数据
        for cls in self.cls:
            self.trad_data[cls] = dict()
            self.new_data[cls] = dict()
            cls_path_trad = osp.join(self.data_path, "{}_trad".format(cls))
            cls_path_new = osp.join(self.data_path, "{}_new".format(cls))

            # 读取传统数据
            for name in os.listdir(cls_path_trad):
                data = np.genfromtxt(osp.join(cls_path_trad, name), delimiter=";")
                data = data[:, ~np.isnan(data).any(axis=0)]
                self.trad_data[cls][name] = data

            # 读取新数据
            for name in os.listdir(cls_path_new):
                data = np.genfromtxt(osp.join(cls_path_new, name))
                data = data[~np.isnan(data).any(axis=1), :]
                self.new_data[cls][name] = data

    # 使用data_statistics装饰器，获取数据
    @data_statistics
    def get_data(self, cls: str = "13DKB", type: str = "1H", test_num: int = 100):
        # 从传统数据中获取与指定类型匹配的键
        trad_type = None
        for key in self.trad_data[cls].keys():
            trad_type = key if key.startswith(type) else trad_type

        # 根据键获取传统数据和新数据
        trad_data = self.trad_data[cls][trad_type]
        new_data = self.new_data[cls][type]
        n = trad_data.shape[0]

        # 生成测试数据索引
        test_idx = np.linspace(0, n - 1, num=test_num, endpoint=True).astype(int)
        train_idx = [i for i in range(n) if i not in test_idx]

        return new_data, trad_data[train_idx], trad_data[test_idx]
