import os
import os.path as osp
import numpy as np


class MideaData(object):
    def __init__(self) -> None:
        self.data_path = "../data/"
        self.data = {
            name.split(".")[0]: np.genfromtxt(osp.join(self.data_path, name))
            for name in os.listdir(self.data_path)
        }

    def get_data(self, name: str = "data_chu_1a", test_num: int = 100):
        n = self.data[name].shape[0]
        test_idx = np.linspace(0, n - 1, num=test_num, endpoint=True).astype(int)
        train_idx = [i for i in range(n) if i not in test_idx]
        return self.data[name][train_idx], self.data[name][test_idx]
