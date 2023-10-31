import os
import os.path as osp
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import time
import copy
from typing import List


# 装饰器，用于输出数据统计信息
def data_statistics(f):
    def wrapper(*args, **kwargs):
        train, test = f(*args, **kwargs)
        print(kwargs)
        print(f"<========= Loading {kwargs['cls']} =========>")
        print(f"train length: {len(train)=}")
        print(f"train shape: {train[0][1].shape=}")
        print(f"test length: {len(test)=}")
        print(f"test shape: {test[0][1].shape=}")
        print(f"<========= Data Loaded =========>")
        time.sleep(3)
        return train, test

    return wrapper


class MideaDataCLS4(object):
    """
    美的数据类，用于处理和存储从不同文件中加载的数据。
    该类分别处理传统数据和新数据，并将它们存储为字典。

    属性:
        data_path (str): 数据文件夹的路径。
        cls (list): 分类名称列表。
        trad_data (dict): 存储传统数据的字典。
        new_data (dict): 存储新数据的字典。

    方法:
        get_data(cls, test_num, seed): 返回经过处理的训练数据和测试数据。
        expand_data(self, data_ls, n): 用于按列进行数据扩充, n为扩充倍数。
    """

    def __init__(self, cls=["13DKB_cls4"]) -> None:
        self.data_path = "../data/"
        self.cls = cls
        self.trad_data = dict()
        self.new_data = dict()
        for cls in self.cls:
            self.trad_data[cls] = dict()
            self.new_data[cls] = dict()
            cls_path_trad = osp.join(self.data_path, "{}_trad".format(cls))
            cls_path_new = osp.join(self.data_path, "{}_new".format(cls))

            for name in sorted(os.listdir(cls_path_trad)):
                if name.startswith("."):
                    continue
                data = np.genfromtxt(osp.join(cls_path_trad, name), delimiter=",")
                data = data[:, ~np.isnan(data).any(axis=0)]
                self.trad_data[cls][name] = data

            # 读取新数据并线性插值，使其对齐传统数据
            for name in sorted(os.listdir(cls_path_new)):
                if name.startswith("."):
                    continue
                data = np.genfromtxt(osp.join(cls_path_new, name), delimiter=",")
                data = data[~np.isnan(data).any(axis=1), :]
                data = data[np.isfinite(data).all(axis=1), :]
                x, y = data[:, 0], data[:, 1]
                if self.cls[0] == "13DKB2" or self.cls[0] == "13DKB_cls4":
                    x /= 1e6
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                x_trad = self.trad_data[cls][name][:, 0]
                y_interpolate = f(x_trad)
                self.new_data[cls][name] = np.c_[x_trad, y_interpolate]
        
        self.ori_trad = copy.deepcopy(self.trad_data)
        self.ori_new = copy.deepcopy(self.new_data)

    # @data_statistics
    def get_data(
        self,
        cls: str = "13DKB",
        test_cls: List[str] = "1H",
        random_dete: bool = False,
        test_num: int = 3,
        seed: int = None,
    ):
        """
        获取指定分类的训练数据和测试数据。

        参数:
            cls (str): 分类名称，默认为 "13DKB"。
            test_num (int): 测试数据集中的数据类型数量，默认为 3。
            seed (int): 随机种子，默认为 None。

        返回:
            train_data (list): 训练数据列表。(20001, 3) 第一列为x, 第二列为y_new, 第三列为y_trad
            test_data (list): 测试数据列表。(20001, 3) 第一列为x, 第二列为y_new, 第三列为y_trad。
            训练目标就是把x, y_new当作输入, y_trad当作输出
        """
        if seed != None:
            random.seed(seed)
        type_list = list(self.trad_data[cls].keys())
        test_type = random.sample(type_list, test_num) if random_dete else test_cls

        if test_cls == "None":
            test_type = []

        train_type = [x for x in type_list if x not in test_type]
        train_data = [
            (
                tr,
                np.c_[
                    self.new_data[cls][tr][:, 0],
                    self.new_data[cls][tr][:, 1],
                    self.trad_data[cls][tr][:, 1],
                ],
            )
            for tr in train_type
        ]
        test_data = [
            (
                te,
                np.c_[
                    self.new_data[cls][te][:, 0],
                    self.new_data[cls][te][:, 1],
                    self.trad_data[cls][te][:, 1],
                ],
            )
            for te in test_type
        ]
        return train_data, test_data
    
    # @data_statistics
    def get_ori_data(
        self,
        cls: str = "13DKB",
        test_cls: List[str] = ["1H"],
        random_dete: bool = False,
        test_num: int = 3,
        seed: int = None,
    ):
        if seed != None:
            random.seed(seed)
        type_list = list(self.trad_data[cls].keys())
        test_type = random.sample(type_list, test_num) if random_dete else test_cls

        if test_cls == "None":
            test_type = []

        train_type = [x for x in type_list if x not in test_type]
        train_data = [
            (
                tr,
                np.c_[
                    self.ori_new[cls][tr][:, 0],
                    self.ori_new[cls][tr][:, 1],
                    self.ori_trad[cls][tr][:, 1],
                ],
            )
            for tr in train_type
        ]
        test_data = [
            (
                te,
                np.c_[
                    self.ori_new[cls][te][:, 0],
                    self.ori_new[cls][te][:, 1],
                    self.ori_trad[cls][te][:, 1],
                ],
            )
            for te in test_type
        ]
        return train_data, test_data

    def expand_data(self, data_ls, n):
        for k in range(len(data_ls)):
            data = data_ls[k][1]
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
            data_ls[k] = (data_ls[k][0], np.array(expanded_data))
        return data_ls

    def plot_data(self, cls="13DKB", name="plot"):
        keys = sorted(list(self.trad_data[cls].keys()), key=lambda x: int(x[:-1]))
        total = len(keys)
        cols = 4
        rows = total // cols + total % cols
        position = range(1, total + 1)

        fig = plt.figure(figsize=(15, 6), dpi=100)

        for k, pos in zip(keys, position):
            ax = fig.add_subplot(rows, cols, pos)
            ax.set_title(k)
            
            if k in self.trad_data[cls]:
                data = self.trad_data[cls][k]
                ax.plot(data[:,0], data[:,1], label='trad_data')
                data = self.new_data[cls][k]
                ax.plot(data[:,0], data[:,1], label='new_data')

            ax.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(f'{name}.png')

