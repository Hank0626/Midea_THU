import os
import os.path as osp
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import time
import copy


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


class MideaData(object):
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

    def __init__(self, cls=["13DKB"]) -> None:
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
                data = np.genfromtxt(osp.join(cls_path_trad, name), delimiter=";")
                data = data[:, ~np.isnan(data).any(axis=0)]
                self.trad_data[cls][name] = data

            # 读取新数据并线性插值，使其对齐传统数据
            for name in sorted(os.listdir(cls_path_new)):
                if name.startswith("."):
                    continue
                data = np.genfromtxt(osp.join(cls_path_new, name))
                data = data[~np.isnan(data).any(axis=1), :]
                data = data[np.isfinite(data).all(axis=1), :]
                x, y = data[:, 0], data[:, 1]
                if self.cls[0] == "13DKB2":
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
        test_cls: str = "1H",
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
        test_type = random.sample(type_list, test_num) if random_dete else [test_cls]

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
        test_cls: str = "1H",
        random_dete: bool = False,
        test_num: int = 3,
        seed: int = None,
    ):
        if seed != None:
            random.seed(seed)
        type_list = list(self.trad_data[cls].keys())
        test_type = random.sample(type_list, test_num) if random_dete else [test_cls]

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

    def get_mu_sigma(self, cat="H", win=7):
        """
        cat: H or V, process H data or V data
        win: window size to decide a abnormal data
        """
        # find abnormal index
        trad = self.trad_data[self.cls[0]][f"2{cat}"]
        new = self.new_data[self.cls[0]][f"2{cat}"]
        half_win = win // 2
        abnormal_index = []
        abnormal_mount = []
        abnormal_delta = []
        for i in range(half_win, new.shape[0] - half_win):
            new_win_data = list(new[i - half_win : i + half_win + 1, 1])
            trad_win_data = list(trad[i - half_win : i + half_win + 1, 1])

            if new_win_data[: half_win + 1] == sorted(
                new_win_data[: half_win + 1]
            ) and new_win_data[half_win:] == sorted(
                new_win_data[half_win:], reverse=True
            ):
                abnormal_index.append(i)

                new_win_height = new_win_data[half_win] - 0.5 * (
                    new_win_data[0] + new_win_data[-1]
                )
                trad_win_height = trad_win_data[half_win] - 0.5 * (
                    trad_win_data[0] + trad_win_data[-1]
                )

                abnormal_mount.append(new_win_height)
                abnormal_delta.append(trad_win_height)

        print(f"cat{cat}win{win}_{len(abnormal_index)}")

        return (
            np.mean(abnormal_mount),
            np.std(abnormal_mount),
            np.mean(abnormal_delta),
            np.std(abnormal_delta),
        )

    def gen_mount_data(self, win=7, num_mount=20, scale = 1.):
        """
            win: window size to decide an abnormal data
            num_mount: nums of generted abnormal data 
            trad_mount_data, new_align_mount_data: data augmented--> {}
        """
        
        def apply_delta_with_peak(data, indices, values, weight_window):
            assert len(indices) == len(values), "indices and delta_values must have the same length"
            assert len(weight_window) % 2 == 1, "weight_window must have an odd length"
            
            half_window_len = len(weight_window) // 2
            for idx, value in zip(indices, values):
                if idx < half_window_len or idx + half_window_len >= len(data):
                    print(f"Warning: index {idx} is too close to the edge of the array, skip...")
                    continue
                data[idx - half_window_len:idx + half_window_len + 1, 1] += value * weight_window
            return data
        
        def generate_weight_window(win, scale):
            assert win % 2 == 1, "win must be an odd number"
            
            half_win = win // 2
            x = np.linspace(-half_win, half_win, win)

            pdf = norm.pdf(x, 0, half_win)

            weight_window = pdf / pdf.max() * scale
            
            return weight_window

        self.win = win
        self.num_mount = num_mount
        data_list = list(range(1, 14))
        data_list.remove(2)
        
        weight_window = generate_weight_window(win, scale)
        print(weight_window)

        for cls in self.cls:
            for cat in ['H', 'V']:
                mount_mean, mount_std, delta_mean, delta_std = self.get_mu_sigma(cat=cat, win=win)
                for i in data_list:
                    new_data = self.new_data[cls][f'{i}{cat}'].copy()
                    trad_data = self.trad_data[cls][f'{i}{cat}'].copy()

                    mount_index = np.random.randint(0, new_data.shape[0], num_mount)
                    mounts_value = np.array([np.abs(np.random.normal(loc=mount_mean, scale=mount_std, size=1)) for _ in range(num_mount)]).squeeze()
                    delta_value = np.array([np.abs(np.random.normal(loc=delta_mean, scale=delta_std, size=1)) for _ in range(num_mount)]).squeeze()
                    # import pdb; pdb.set_trace()
                    self.new_data[cls][f'{i}{cat}'] = apply_delta_with_peak(new_data, mount_index, mounts_value, weight_window)
                    self.trad_data[cls][f'{i}{cat}'] = apply_delta_with_peak(trad_data, mount_index, delta_value, weight_window)

    def plot_data(self, cls="13DKB", name="plot"):
        keys = sorted(list(self.trad_data[cls].keys()), key=lambda x: int(x[:-1]))

        total = len(keys)
        cols = 4
        rows = total // cols
        rows += total % cols
        position = range(1, total + 1)

        fig = plt.figure(figsize=(30, 30))

        for k, pos in zip(keys, position):
            ax = fig.add_subplot(rows, cols, pos)
            ax.set_title(k)
            
            if k in self.trad_data['13DKB']:
                data = self.trad_data['13DKB'][k]
                ax.plot(data[:,0], data[:,1], label='trad_data')
                data = self.new_data['13DKB'][k]
                ax.plot(data[:,0], data[:,1], label='new_data')

            ax.legend()

        plt.tight_layout()
        plt.savefig(f'{name}.png')
        

if __name__ == "__main__":
    data = MideaData()
    data.gen_mount_data(win=9, scale=2.)
    import pdb

    pdb.set_trace()
