import numpy as np
import matplotlib.pyplot as plt
from utils.evaluate import np_mae, np_mape, np_rmse
from sklearn import svm
from bayes_opt import BayesianOptimization
from Midea_THU.src.lgm_data import DataAlig, MideaData
import pickle
from sklearnex import patch_sklearn, unpatch_sklearn
import logging
import sys
import lightgbm as lgb
import time
import os
import warnings

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
# def train_svr(train_in,train_t,test_in,test_t,C,epsilon,gamma, path):

#     model = svm.SVR(degree=3, gamma=gamma, coef0=0.0, tol=0.005, C=C, epsilon=epsilon, shrinking=True,
#             cache_size=200, verbose=False, max_iter=- 1)

#     model.fit(train_in, train_t)
#     pre = model.predict(test_in)
#     val = -np_mae(pre,test_t)
#     # 保存Model
#     name = str(C)+'.pkl'
#     with open(path+name, "wb") as f:
#         pickle.dump(model, f)
#     # 读取Model
#     # clf_1 = joblib.load('./clf.pkl')
#     # metrics = [np_mae, np_rmse, np_mape, np_mmae]
#     #
#     # print("metrics1:\t[mae | rmse | mape | mmae]")
#     # print("results1:\t", [np.round(f(pre, test_t), 3) for f in metrics])
#     return -np_mae(pre,test_t)


def train_lightgbm(train_in, train_t, test_in, test_t, path, **lgm_paras):
    # model = svm.SVR(degree=3, gamma=gamma, coef0=0.0, tol=0.005, C=C, epsilon=epsilon, shrinking=True,
    #         cache_size=200, verbose=False, max_iter=- 1)
    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting": "goss",
        "learning_rate": lgm_paras["learning_rate"],
        "num_leaves": lgm_paras["num_leaves"],
        "n_estimators": lgm_paras["n_estimators"],
        "max_depth": lgm_paras["max_depth"],
        "num_threads": 16,
        "device": "gpu",
        "gpu_platform_id": 0,
        "verbose": -1,
    }
    train_data = lgb.Dataset(train_in, label=train_t)
    test_data = lgb.Dataset(test_in, label=test_t)
    model = lgb.train(params, train_data)
    pre = model.predict(test_in)

    val = -np_mae(pre, test_t)
    # 保存Model
    name = "model.pkl"
    with open(path + name, "wb") as f:
        pickle.dump(model, f)
    # 读取Model
    # clf_1 = joblib.load('./clf.pkl')
    # metrics = [np_mae, np_rmse, np_mape, np_mmae]
    #
    # print("metrics1:\t[mae | rmse | mape | mmae]")
    # print("results1:\t", [np.round(f(pre, test_t), 3) for f in metrics])
    return -np_mae(pre, test_t)


class Bayes(object):
    def __init__(self, test_name="13V", k=5, range=True, needx=True) -> None:
        self.data = MideaData()
        self.train_in = []
        self.train_tar = []
        self.test_in = []
        self.test_tar = []
        self.param = []
        self.test_name = test_name
        self.dataalig = DataAlig(k, needrange=range, needx=needx)
        self.str_list = ["H", "V"]
        self.exp_path = "./experiments/"
        if range:
            self.exp_path = self.exp_path + "range/"
        else:
            self.exp_path = self.exp_path + "norange/"

        self.logger = logging.getLogger()

        self.logger.setLevel(logging.INFO)

        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(self.exp_path + "test.log")

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()

        # 定义handler的输出格式formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        # logger.addFilter(filter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 记录一条日志
        self.logger.info("start")

    def get_data(self, cls="13DKB"):
        cnt = 0
        for i in range(1, 14):
            one_hot = np.zeros(13)
            one_hot[i - 1] = 1
            for s in self.str_list:
                name = str(i) + s
                new, trad, _ = self.data.get_data(cls=cls, type=name, ratio=0)
                new[:, 0] /= 1e6
                trad[:, 0] /= 1e6
                input_t, target_t = self.dataalig.pros_data(new, trad)
                input_oh = np.tile(one_hot, (input_t.shape[0], 1))
                input_t = np.concatenate((input_t, input_oh), axis=1)
                if name == self.test_name:
                    self.test_in = input_t
                    self.test_tar = target_t
                elif cnt == 0:
                    self.train_in = input_t
                    self.train_tar = target_t
                    cnt += 1
                else:
                    self.train_in = np.concatenate((self.train_in, input_t), axis=0)
                    self.train_tar = np.concatenate((self.train_tar, target_t), axis=0)
                    cnt += 1

    def rf_cv(self, **lgm_paras):
        start = time.time()
        lgm_paras = {
            "learning_rate": lgm_paras["learning_rate"],
            "num_leaves": int(lgm_paras["num_leaves"]),
            "n_estimators": int(lgm_paras["n_estimators"]),
            "max_depth": int(lgm_paras["max_depth"]),
        }
        # val = train_svr(self.train_in,self.train_tar,self.test_in,self.test_tar,C,epsilon,gamma,self.exp_path)
        val = train_lightgbm(
            self.train_in,
            self.train_tar,
            self.test_in,
            self.test_tar,
            self.exp_path,
            **lgm_paras
        )
        end = time.time()
        t = end - start
        print("time", t, "s")
        # self.logger.info('[train] mae = %.4f \t C = %f \t epsilon = %f \t gamma = %f' % (-val, C,epsilon,gamma))
        return val

    def opt(self):
        self.get_data()
        print(self.train_in.shape)
        print(self.train_tar.shape)
        print(self.test_in.shape)
        print(self.test_tar.shape)
        print(self.train_in[20])
        print(self.train_tar[20])

        patch_sklearn()
        rf_bo = BayesianOptimization(
            self.rf_cv,
            {
                "learning_rate": (0.001, 1),
                "num_leaves": (10, 20),
                "n_estimators": (100, 300),
                "max_depth": (5, 10),
            },
        )
        rf_bo.maximize()
        lst = []
        for i in rf_bo.res:
            lst.append(i["target"])
        idx = lst.index(max(lst))
        best = rf_bo.res[idx]
        print(best)
        self.param = best["params"]
        # self.train_best()

    def train_best(self):
        start = time.time()
        C = self.param["C"]
        epsilon = self.param["epsilon"]
        gamma = self.param["gamma"]
        # C ,epsilon,gamma = self.param
        model = svm.SVR(
            degree=3,
            gamma=gamma,
            coef0=0.0,
            tol=0.005,
            C=C,
            epsilon=epsilon,
            shrinking=True,
            cache_size=200,
            verbose=False,
            max_iter=-1,
        )
        name = str(C) + ".pkl"
        with open(self.exp_path + "8.921149744883394.pkl", "rb") as f:
            model = pickle.load(f)

        pre = model.predict(self.test_in)
        # metrics = [np_mae, np_rmse, np_mape, np_mmae]
        metrics = [np_mae, np_rmse, np_mape]
        print("metrics1:\t[mae | rmse | mape | mmae]")
        res = [np.round(f(pre, self.test_tar), 3) for f in metrics]
        print("results1:\t", res)
        self.logger.info(
            "[best] C = %f \t epsilon = %f \t gamma = %f" % (C, epsilon, gamma)
        )
        self.logger.info("metrics1:\t[mae | rmse | mape | mmae]")
        self.logger.info("results1:\t", res)

        end = time.time()
        plt.plot(self.test_in[:, 0], pre, label="pre")
        plt.plot(self.test_in[:, 0], self.test_tar, label="gt")
        # plt.plot(test[:, 0], pre2, label="pre2")
        # plt.plot(test[:, 0], pre3, label="pre3")
        # plt.plot(test[:, 0], pre4, label="pre4")
        # plt.plot(train[:, 0], train[:, 1]-train[:, 2], label="sub")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test_name = [str(i) + "H" for i in range(1, 11)] + [str(i) + "V" for i in range(1, 11)]
    print(test_name)
    for test in test_name:
        bys = Bayes(test_name=test, k=101, range=True, needx=True)
        bys.opt()
        print('-------------------------', test, '-----------------------')

# 

"""
1. 

{"learning_rate": (0.001, 1),
'num_leaves': (10, 100),
'n_estimators':(100, 2000),
'max_depth': (5, 20),
}
1. range True, needx True  1.288 k = 5

{"learning_rate": (0.001, 1),
'num_leaves': (10, 50),
'n_estimators':(100, 500),
'max_depth': (5, 10),
}
1.26

{"learning_rate": (0.001, 0.1),
'num_leaves': (10, 50),
'n_estimators':(100, 500),
'max_depth': (5, 10),
}1.31


"learning_rate": (0.001, 0.1),
"num_leaves": (10, 20),
"n_estimators": (100, 300),
"max_depth": (5, 10),
1.31

------------------final hyper-params--------------------
{
    "learning_rate": (0.001, 1),
    "num_leaves": (10, 20),
    "n_estimators": (100, 300),
    "max_depth": (5, 10),
},
1. False True k=5 1.2756
2. False True k=11 1.2655
3. False True k=21 1.2404
3. False True k=31 1.2367
4. False True k=51 1.2161
5. False True k=101 1.1751
6. True True k=101 1.1625
-----------不同數據集試試-- ----------------
True True 101 
13H 1.1772
12H 1.1924
12V 1.2190
11V 1.2868
11H 1.3422
"""
