<<<<<<< HEAD

import numpy as np

# 使用未对其的原始数据
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
        print(f"new shape: {new.shape}")  # 输出新设备数据的形状
        print(f"train shape: {train.shape}")  # 输出传统设备训练数据的形状
        print(f"test shape: {test.shape}")  # 输出传统设备测试数据的形状
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
                data = np.genfromtxt(osp.join(cls_path_trad, name), delimiter=";",dtype="float64")
                data = data[:, ~np.isnan(data).any(axis=0)]
                self.trad_data[cls][name] = data

            # 读取新数据
            for name in os.listdir(cls_path_new):
                data = np.genfromtxt(osp.join(cls_path_new, name),dtype="float64")
                data = data[~np.isnan(data).any(axis=1), :]
                data = data[~np.isneginf(data).any(axis=1), :]
                self.new_data[cls][name] = data

    # @data_statistics
    def get_data(self, cls: str = "13DKB", type: str = "1H", ratio: float = 0.25):
        """
        参数:
            cls (str): 设备型号. Defaults to "13DKB".
            type (str): 测试类别. Defaults to "1H".
            ratio (ratio): 测试集比例. Defaults to 0.25.

        Returns:
            新设备的所有数据(16167,2), 旧设备训练数据, 旧设备测试数据
        """

        # 从传统数据中获取与指定类型匹配的键
        trad_type = None
        for key in self.trad_data[cls].keys():
            trad_type = key if key.startswith(type) else trad_type

        # 根据键获取传统数据和新数据
        trad_data = self.trad_data[cls][trad_type]
        new_data = self.new_data[cls][type]
        n = trad_data.shape[0]

        # 生成测试数据索引
        test_idx = np.linspace(0, n - 1, num=int(ratio * n), endpoint=True).astype(int)
        train_idx = [i for i in range(n) if i not in test_idx]

        return new_data, trad_data[train_idx], trad_data[test_idx]

class DataAlig(object):

    def __init__(self, k=5, needx=True, needrange=True) -> None:

        self.k = k #最邻近的k个点
        self.needx = needx
        self.needrange = needrange

    # @data_statistics
    def pros_data(self, new, trad):
        """
        参数:
            new (N,2): 新设备数据
            trad (M,2): 旧设备数据

        Returns:
            input (M,k+1): 旧设备各频率对应新设备的最近K个数据+频率
            target (M,1): 旧设备数据
        """

        # 从传统数据中获取与指定类型匹配的键
        N = new.shape[0]
        M = trad.shape[0]
        input = np.zeros((M,self.k))
        if self.needx :
            input = np.concatenate((input, input),axis=1)
        target = trad[:,1]
        R = np.zeros(M)
        j=0
        for i in range(0,M) :
            x = trad[i,0]
            # 寻找最近点
            while j < N-1 and np.abs(new[j,0]-x)>np.abs(new[j+1,0]-x):
                j = j+1
            # 最近k个
            if x<new[j,0]:
                start = j - int(self.k/2)
                end = start + self.k
            else:
                end = j + int(self.k / 2)+1
                start = end - self.k

            if start<0:
                start = 0
                end = self.k
            elif end>N:
                end = N
                start = N-self.k

            # 计算极差
            kmax = max(new[start:end, 1])
            kmin = min(new[start:end, 1])
            R[i] = kmax-kmin

            if self.needx:
                list = new[start:end,:].transpose().reshape(-1)
            else :
                list = new[start:end, 2].reshape(-1)
            input[i] = list

        input = np.concatenate((input, trad[:,0].reshape(-1,1)),axis=1)

        if self.needrange :
            input = np.concatenate((input, R.reshape(-1, 1)), axis=1)

        return input, target
=======

import numpy as np

# 使用未对其的原始数据

class DataAlig(object):

    def __init__(self, k=5, needx=True, needrange=True) -> None:
        self.k = k #最邻近的k个点
        self.needx = needx
        self.needrange = needrange

    # @data_statistics
    def pros_data(self, new, trad):
        """
        参数:
            new (N,2): 新设备数据
            trad (M,2): 旧设备数据

        Returns:
            input (M,k+1): 旧设备各频率对应新设备的最近K个数据+频率
            target (M,1): 旧设备数据
        """

        # 从传统数据中获取与指定类型匹配的键
        N = new.shape[0]
        M = trad.shape[0]
        input = np.zeros((M,self.k))
        if self.needx :
            input = np.concatenate((input, input),axis=1)
        target = trad[:,1]
        R = np.zeros(M)
        j=0
        for i in range(0,M) :
            x = trad[i,0]
            # 寻找最近点
            while j < N-1 and np.abs(new[j,0]-x)>np.abs(new[j+1,0]-x):
                j = j+1
            # 最近k个
            if x<new[j,0]:
                start = j - int(self.k/2)
                end = start + self.k
            else:
                end = j + int(self.k / 2)+1
                start = end - self.k

            if start<0:
                start = 0
                end = self.k
            elif end>N:
                end = N
                start = N-self.k

            # 计算极差
            kmax = max(new[start:end, 1])
            kmin = min(new[start:end, 1])
            R[i] = kmax-kmin

            if self.needx:
                list = new[start:end,:].transpose().reshape(-1)
            else :
                list = new[start:end, 2].reshape(-1)
            input[i] = list

        input = np.concatenate((input, trad[:,0].reshape(-1,1)),axis=1)

        if self.needrange :
            input = np.concatenate((input, R.reshape(-1, 1)), axis=1)

        return input, target
>>>>>>> c78c412de2900af3ebb3032bd4f1c4e97c112707
