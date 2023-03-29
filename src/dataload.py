import os
import os.path as osp
import numpy as np
import time


def data_statistics(f):
    def wrapper(*args, **kwargs):
        train, test = f(*args, **kwargs)
        print(f"<========= Loading {kwargs['name']} =========>")
        print(f"train shape: {train.shape=}")
        print(f"test shape: {test.shape=}")
        print(f"<========= Data Loaded =========>")
        time.sleep(3)
        return train, test

    return wrapper


class MideaData(object):
    def __init__(self) -> None:
        self.data_path = "../data/"
        self.data = {
            name.split(".")[0]: np.genfromtxt(osp.join(self.data_path, name))
            for name in os.listdir(self.data_path)
        }

    @data_statistics
    def get_data(self, name: str = "data_chu_1a", test_num: int = 100):
        data = self.data[name]
        data[:, 0] /= 1e6
        n = self.data[name].shape[0]
        test_idx = np.linspace(0, n - 1, num=test_num, endpoint=True).astype(int)
        train_idx = [i for i in range(n) if i not in test_idx]

        return data[train_idx], data[test_idx]
