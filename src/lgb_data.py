import numpy as np
import matplotlib.pyplot as plt
# 使用未对其的原始数据
import os
import os.path as osp
import numpy as np
import time
from scipy import interpolate

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

    def __init__(self, align, scale=1.0) -> None:
        """
            align: aling data or not (from 16000 to 20000)
        """
        self.data_path = "/data/lulongfei/thucode/meidi/Midea_THU/data_1"
        self.cls = ["13DKB"]
        self.trad_data = dict()  # 传统设备数据
        self.new_data = dict()  # 新设备数据
        self.scale = scale
        for cls in self.cls:
            self.trad_data[cls] = dict()
            self.new_data[cls] = dict()
            cls_path_trad = osp.join(self.data_path, "{}_trad".format(cls))
            cls_path_new = osp.join(self.data_path, "{}_new".format(cls))
            cls_path_new_align = osp.join(self.data_path, "{}_new_align".format(cls))

            # 读取传统数据
            for name in os.listdir(cls_path_trad):
                data = np.genfromtxt(
                    osp.join(cls_path_trad, name), delimiter=";", dtype="float64"
                )
                data = data[:, ~np.isnan(data).any(axis=0)]
                self.trad_data[cls][name] = data
            # 读取新数据
            for name in os.listdir(cls_path_new):
                data = np.genfromtxt(osp.join(cls_path_new, name), dtype="float64")
                data = data[~np.isnan(data).any(axis=1), :]
                data = data[~np.isneginf(data).any(axis=1), :]
                self.new_data[cls][name] = data
            # 插值对齐新数据
            self.new_align_data = {}
            self.new_align_data[cls] = {}
            # data align and save data for latter use
            if align:
                new_align_root = osp.join(self.data_path, "{}_new_align".format(cls))
                if not os.path.exists(new_align_root):
                    os.mkdir(new_align_root)
                
                x_new = np.linspace(30000000, 1000000000, 20001)
                for k, data in zip(self.new_data[cls].keys() ,self.new_data[cls].values()):
                    xi = np.concatenate([data[:, 0], np.array([1000000000])])  # 1000000000.000   -3.952
                    yi = np.concatenate([data[:, 1], np.array([data[-1, 1]])])
                    # x_new = trad[:, 0]
                    interp = interpolate.interp1d(xi, yi, kind = "cubic")
                    y_new = interp(x_new)
                    new_align = np.stack([x_new, y_new], 1)
                    self.new_align_data[cls][k] = new_align
                    file_path = osp.join(self.data_path, "{}_new_align".format(cls), k)
                    np.savetxt(file_path, new_align, fmt='%f', delimiter=';')
            else:  # read saved align data
                for name in os.listdir(cls_path_new_align):
                    data = np.genfromtxt(
                        osp.join(cls_path_new_align, name), delimiter=";", dtype="float64"
                    )
                    data = data[:, ~np.isnan(data).any(axis=0)]
                    self.new_align_data[cls][name] = data
                # self.new_data = self.new_align_data

    def get_mu_sigma(self, cat='H', win=7): 
            """
                cat: H or V, process H data or V data
                win: window size to decide a abnormal data
            """
            # find abnormal index
            trad =  self.trad_data[self.cls[0]]['2{}'.format(cat)]
            new = self.new_align_data[self.cls[0]]['2{}'.format(cat)]
            half_win = win // 2
            abnormal_index = []  
            n, _ = new.shape
            abnormal_mount = []
            abnormal_delta = []
            for i in range(half_win, n - half_win):
                new_win_data = list(new[i - half_win:i + half_win + 1, 1])
                trad_win_data = list(trad[i - half_win:i + half_win + 1, 1])

                if new_win_data[:half_win + 1] == sorted(new_win_data[:half_win + 1]) and new_win_data[half_win:] == sorted(new_win_data[half_win:], reverse=True):
                    abnormal_index.append(i)
                    
                    new_win_height = new_win_data[half_win] - 0.5 * (new_win_data[0] + new_win_data[-1])
                    trad_win_height = trad_win_data[half_win] - 0.5 * (trad_win_data[0] + trad_win_data[-1])

                    abnormal_mount.append(new_win_height)
                    abnormal_delta.append(trad_win_height)  
                    # 这里现在先用一个点的值，因为后面突变也是只突变了一个数据点

            print(f'cat{cat}win{win}_{len(abnormal_index)}')
            # abnormal_mount = new[abnormal_index, 1]  # abnormal value
            # abnormal_delta = trad[abnormal_index, 1] - new[abnormal_index, 1]  # abnormal delta y
            mount_mean, mount_std = np.mean(abnormal_mount), np.std(abnormal_mount)
            abnormal_delta_mean, abnormal_delta_std = np.mean(abnormal_delta), np.std(abnormal_delta)
            return mount_mean, mount_std, abnormal_delta_mean, abnormal_delta_std

    def gen_mount_data(self, win=7, num_mount=20):
        """
            win: window size to decide an abnormal data
            num_mount: nums of generted abnormal data 
            trad_mount_data, new_align_mount_data: data augmented--> {}
        """
        self.win = win
        self.num_mount = num_mount
        data_list = list(range(1, 14))
        data_list.remove(2)
        self.new_align_mount_data = {}
        self.trad_mount_data = {}
        for cls in self.cls:
            self.new_align_mount_data[cls] = {}
            self.trad_mount_data[cls] = {}
            for cat in ['H', 'V']:
                mount_mean, mount_std, delta_mean, delta_std = self.get_mu_sigma(cat=cat, win=win)
                # print(cat, mount_mean, mount_std, delta_mean, delta_std)
                for i in data_list:
                    new_data = self.new_align_data[cls]['{}{}'.format(i, cat)].copy()
                    trad_data = self.trad_data[cls]['{}{}'.format(i, cat)].copy()
                    num_new_data = new_data.shape[0]

                    mount_index = np.random.randint(0, num_new_data, num_mount)
                    mounts_value = []
                    delta_value = []
                    while len(mounts_value) < num_mount:
                        value = np.random.normal(loc=mount_mean, scale=mount_std, size=1)
                        if value > 0:
                            mounts_value.append(value)
                    
                    while len(delta_value) < num_mount:
                        value = np.random.normal(loc=delta_mean, scale=delta_std, size=1)
                        if value > 0:
                            delta_value.append(value)
                    mounts_value = np.array(mounts_value).squeeze() * self.scale
                    delta_value = np.array(delta_value).squeeze() * self.scale
                    
                    # print(i, cat)
                    # print(f"mount_index:{mount_index}")
                    # print(f"mounts_value:{mounts_value}")
                    # print(f'delta_value:{delta_value}')
                    # mounts_value = np.random.normal(loc=mount_mean, scale=mount_std, size=num_mount)
                    # delta_value = np.random.normal(loc=delta_mean, scale=delta_std, size=num_mount)

                    new_data[mount_index, 1] = mounts_value + new_data[mount_index, 1]
                    trad_data[mount_index, 1] = trad_data[mount_index, 1] + delta_value
                    self.new_align_mount_data[cls]['{}{}'.format(i, cat)] = new_data
                    self.trad_mount_data[cls]['{}{}'.format(i, cat)] = trad_data
            # 2号数据不做增广，用原始数据替换
            self.trad_mount_data[cls]['2H'] = self.trad_data[cls]['2H']
            self.trad_mount_data[cls]['2V'] = self.trad_data[cls]['2V']
            self.new_align_mount_data[cls]['2H'] = self.new_align_data[cls]['2H']
            self.new_align_mount_data[cls]['2V'] = self.new_align_data[cls]['2V']
            # save aligned mount data
            new_mount_dir = osp.join(self.data_path, "{}_new_align_mount_abs_scale_{}_num_mount_{}".format(cls, self.scale, num_mount))
            trad_mount_dir = osp.join(self.data_path, "{}_trad_align_mount_abs_scale_{}_num_mount{}".format(cls, self.scale, num_mount))
            if not os.path.exists(new_mount_dir):
                os.mkdir(new_mount_dir)
            if not os.path.exists(trad_mount_dir):
                os.mkdir(trad_mount_dir)
            for k, data in zip(self.new_align_mount_data[cls].keys(), self.new_align_mount_data[cls].values()):
                file_path = osp.join(new_mount_dir, k)
                np.savetxt(file_path, data, fmt='%f', delimiter=';')
            
            for k, data in zip(self.trad_mount_data[cls].keys(), self.trad_mount_data[cls].values()):
                file_path = osp.join(trad_mount_dir, k)
                np.savetxt(file_path, data, fmt='%f', delimiter=';')
    

    def plot_data(self):
        img_dir = '/data/lulongfei/thucode/meidi/Midea_THU/data_1/img'
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        data_list = list(range(1, 14))
        # data_list.remove(2)
        for cat in ['H', 'V']:
            for i in data_list:
                fig, axs = plt.subplots(2, 2, figsize=(12, 3))
                # x = list(range(20001))
                y1 = self.new_data[self.cls[0]]['2{}'.format(cat)] # 2号数据
                y2 = self.new_data[self.cls[0]]['{}{}'.format(i, cat)] # x号数据
                y3 = self.new_align_data[self.cls[0]]['{}{}'.format(i, cat)] # x号插值数据
                y4 = self.new_align_mount_data[self.cls[0]]['{}{}'.format(i, cat)] # x号构造异常值的数据

                y1_trad = self.trad_data[self.cls[0]]['2{}'.format(cat)] # 2号数据
                y2_trad = self.trad_data[self.cls[0]]['{}{}'.format(i, cat)] # x号数据
                y3_trad = self.trad_data[self.cls[0]]['{}{}'.format(i, cat)] # x号插值数据
                y4_trad = self.trad_mount_data[self.cls[0]]['{}{}'.format(i, cat)] # x号构造异常值的数据


                # 绘制第一个散点图
                axs[0, 0].plot(list(range(y1.shape[0])), y1[:, 1], c='y', label='new')
                axs[0, 0].plot(list(range(y1_trad.shape[0])), y1_trad[:, 1], c='b', label='trad')
                axs[0, 0].set_title(f'2{cat}')
                # axs[0, 0].legend()

                # 绘制第二个散点图
                axs[0, 1].plot(list(range(y2.shape[0])), y2[:, 1], c='y', label='new')
                axs[0, 1].plot(list(range(y2_trad.shape[0])), y2_trad[:, 1], c='b', label='trad')
                axs[0, 1].set_title(f'{i}{cat}')
                axs[0, 1].legend(loc='lower right')
                # 绘制第三个散点图
                axs[1, 0].plot(list(range(y3.shape[0])), y3[:, 1], c='y', label='new')
                axs[1, 0].plot(list(range(y3_trad.shape[0])), y3_trad[:, 1], c='b', label='trad')
                axs[1, 0].set_title(f'{i}{cat}_cubic')

                # 绘制第四个散点图
                axs[1, 1].plot(list(range(y4.shape[0])), y4[:, 1], c='y', label='new', )
                axs[1, 1].plot(list(range(y4_trad.shape[0])), y4_trad[:, 1], c='b', label='trad')
                axs[1, 1].set_title(f'{i}{cat}_cubic_win{self.win}_nums{self.num_mount}')

                # 调整子图之间的间距
                plt.tight_layout()

                # 显示图形
                plt.show()
                plt.savefig(os.path.join(img_dir, f'{i}{cat}_win{self.win}_num_count{self.num_mount}_abs_scale_{self.scale}.png'))

    # def get_data(self, cls: str = "13DKB", test: int = 1, test_range=[]):
    #     """
    #     参数:
    #         cls (str): 设备型号. Defaults to "13DKB".
    #         type (str): 测试类别. Defaults to "1H".
    #         ratio (ratio): 测试集比例. Defaults to 0.25.

    #     Returns:
    #         新设备的所有数据(16167,2), 旧设备训练数据, 旧设备测试数据
    #     """


    #     # 从传统数据中获取与指定类型匹配的键
    #     # train 12 * 2
    #     self.new_align_data = self.new_align_mount_data
    #     train_new = []
    #     train_trad = []
    #     test_new = []
    #     test_trad = []
    #     for k, v in zip(self.trad_data[cls].keys(), self.trad_data[cls].values()):
    #         if int(k[:-1]) == test:
    #             test_trad.append(v[test_range[0]:test_range[1], ])
    #         else:
    #             train_trad.append(v)
    #     for k, v in zip(self.new_align_data[cls].keys(), self.new_align_data[cls].values()):
    #         if int(k[:-1]) == test:
    #             test_new.append(v[test_range[0]:test_range[1],])
    #         else:
    #             train_new.append(v)
            
    #     train_new = np.concatenate(train_new)
    #     test_new = np.concatenate(test_new)

    #     train_trad = np.concatenate(train_trad)
    #     test_trad = np.concatenate(test_trad)

    #     return train_new, train_trad, test_new, test_trad
    
    def get_data(self, cls: str = "13DKB", test: int = 1, test_range=[]):
        """
        参数:
            cls (str): 设备型号. Defaults to "13DKB".
            type (str): 测试类别. Defaults to "1H".
            ratio (ratio): 测试集比例. Defaults to 0.25.

        Returns:
            新设备的所有数据(16167,2), 旧设备训练数据, 旧设备测试数据
        """


        # 从传统数据中获取与指定类型匹配的键
        # train 12 * 2
        # self.new_align_data = self.new_align_mount_data
        train_new = []
        train_trad = []
        test_new = []
        test_trad = []
        for k in self.trad_data[cls].keys():
            if int(k[:-1]) == test:
                test_trad.append(self.trad_data[cls][k][test_range[0]:test_range[1], ])
            else:
                train_trad.append(self.trad_mount_data[cls][k])

        for k in self.new_align_data[cls].keys():
            if int(k[:-1]) == test:
                test_new.append(self.new_align_data[cls][k][test_range[0]:test_range[1],])
            else:
                train_new.append(self.new_align_mount_data[cls][k])
            
        train_new = np.concatenate(train_new)
        test_new = np.concatenate(test_new)

        train_trad = np.concatenate(train_trad)
        test_trad = np.concatenate(test_trad)

        return train_new, train_trad, test_new, test_trad

if __name__ == '__main__':
    data = MideaData(align=False)
    data.gen_mount_data(win=7, num_mount=20)
    data.plot_data()

# test