<<<<<<< HEAD
import numpy as np
# import tensorflow as tf
# import gpflow
import matplotlib.pyplot as plt
from utils.evaluate import np_mae, np_mape, np_rmse, np_mmae
import click
from sklearn import svm
from bayes_opt import BayesianOptimization
from SVR_data import DataAlig, MideaData
from sklearn.externals import joblib
import pickle
from sklearnex import patch_sklearn, unpatch_sklearn

import time


def train_svr(train_in,train_t,test_in,test_t,C,epsilon,gamma, path='./experiments/'):
    model = svm.SVR(degree=3, gamma=gamma, coef0=0.0, tol=0.005, C=C, epsilon=epsilon, shrinking=True,
            cache_size=200, verbose=False, max_iter=- 1)

    model.fit(train_in, train_t)
    pre = model.predict(test_in)
    val = -np_mae(pre,test_t)
    # 保存Model
    name = str(C)+'.pkl'
    with open(path+name, "wb") as f:
        pickle.dump(model, f)
    # 读取Model
    # clf_1 = joblib.load('./clf.pkl')
    # metrics = [np_mae, np_rmse, np_mape, np_mmae]
    #
    # print("metrics1:\t[mae | rmse | mape | mmae]")
    # print("results1:\t", [np.round(f(pre, test_t), 3) for f in metrics])
    return -np_mae(pre,test_t)


class Bayes(object):
    def __init__(self,test_name = "13V", k=5) -> None:
        self.data = MideaData()
        self.train_in = []
        self.train_tar = []
        self.test_in = []
        self.test_tar = []
        self.param = []
        self.test_name = test_name
        self.dataalig = DataAlig(k)
        self.str_list = ['H','V']

    def get_data(self,cls="13DKB"):
        cnt = 0
        for i in range(1, 14):
            one_hot = np.zeros(13)
            one_hot[i-1] = 1
            for s in self.str_list:
                name = str(i) + s
                new, trad, _ = self.data.get_data(cls=cls, type=name, ratio=0)
                new[:, 0] /= 1e6
                trad[:, 0] /= 1e6
                input_t, target_t = self.dataalig.pros_data(new, trad)
                input_oh = np.tile(one_hot,(input_t.shape[0],1))
                input_t = np.concatenate((input_t, input_oh), axis=1)
                if name == self.test_name:
                    self.test_in = input_t
                    self.test_tar = target_t
                elif cnt == 0:
                    self.train_in = input_t
                    self.train_tar = target_t
                    cnt +=1
                else :
                    self.train_in = np.concatenate((self.train_in, input_t), axis=0)
                    self.train_tar = np.concatenate((self.train_tar, target_t), axis=0)
                    cnt += 1





    def rf_cv(self, C,epsilon,gamma):
        start = time.time()
        val = train_svr(self.train_in,self.train_tar,self.test_in,self.test_tar,C,epsilon,gamma)
        end = time.time()
        t = end-start
        print('time',t,'s')
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
            {'C': (1, 15),
             'epsilon': (0, 5),
             'gamma': (0.1, 100)}
        )
        rf_bo.maximize()
        lst = []
        for i in rf_bo.res:
            lst.append(i['target'])
        idx = lst.index(max(lst))
        best = rf_bo.res[idx]
        print(best)
        self.param = best['params']
        self.train_best()

    def train_best(self, path='./experiments/'):
        start = time.time()
        C = self.param['C']
        epsilon = self.param['epsilon']
        gamma = self.param['gamma']
        # C ,epsilon,gamma = self.param
        model = svm.SVR(degree=3, gamma=gamma, coef0=0.0, tol=0.005, C=C, epsilon=epsilon, shrinking=True,
                        cache_size=200, verbose=False, max_iter=- 1)
        name = str(C) + '.pkl'
        with open(path + name, "r") as f:
            model = pickle.load(f)


        model.fit(self.train_in, self.train_tar)
        pre = model.predict(self.test_in)
        metrics = [np_mae, np_rmse, np_mape, np_mmae]

        print("metrics1:\t[mae | rmse | mape | mmae]")
        print("results1:\t", [np.round(f(pre, self.test_tar), 3) for f in metrics])
        end = time.time()
        plt.title(self.type)
        plt.plot(self.test_in[:, 0], pre, label="pre")
        plt.plot(self.test_in[:, 0], self.test_tar, label="gt")
        # plt.plot(test[:, 0], pre2, label="pre2")
        # plt.plot(test[:, 0], pre3, label="pre3")
        # plt.plot(test[:, 0], pre4, label="pre4")
        # plt.plot(train[:, 0], train[:, 1]-train[:, 2], label="sub")
        plt.legend()
        plt.show()




if __name__ == "__main__":
    bys = Bayes(test_name = "13V",k=5)
    bys.opt()
=======
import numpy as np
# import tensorflow as tf
# import gpflow
import matplotlib.pyplot as plt
from utils.mideadata import MideaData
from utils.evaluate import np_mae, np_mape, np_rmse, np_mmae
import click
from sklearn import svm
from bayes_opt import BayesianOptimization
from SVR_data import DataAlig
from sklearn.externals import joblib
import pickle
from sklearnex import patch_sklearn, unpatch_sklearn

import time


def train_svr(train_in,train_t,test_in,test_t,C,epsilon,gamma, path='./experiments/'):
    model = svm.SVR(degree=3, gamma=gamma, coef0=0.0, tol=0.005, C=C, epsilon=epsilon, shrinking=True,
            cache_size=200, verbose=False, max_iter=- 1)

    model.fit(train_in, train_t)
    pre = model.predict(test_in)
    val = -np_mae(pre,test_t)
    # 保存Model
    name = str(C)+'.pkl'
    with open(path+name, "wb") as f:
        pickle.dump(model, f)
    # 读取Model
    # clf_1 = joblib.load('./clf.pkl')
    # metrics = [np_mae, np_rmse, np_mape, np_mmae]
    #
    # print("metrics1:\t[mae | rmse | mape | mmae]")
    # print("results1:\t", [np.round(f(pre, test_t), 3) for f in metrics])
    return -np_mae(pre,test_t)


class Bayes(object):
    def __init__(self,test_name = "13V", k=5) -> None:
        self.data = MideaData()
        self.train_in = []
        self.train_tar = []
        self.test_in = []
        self.test_tar = []
        self.param = []
        self.test_name = test_name
        self.dataalig = DataAlig(k)
        self.str_list = ['H','V']

    def get_data(self,cls="13DKB"):
        cnt = 0
        for i in range(1, 14):
            one_hot = np.zeros(13)
            one_hot[i-1] = 1
            for s in self.str_list:
                name = str(i) + s
                new, trad, _ = self.data.get_data(cls=cls, type=name, ratio=0)
                new[:, 0] /= 1e6
                trad[:, 0] /= 1e6
                input_t, target_t = self.dataalig.pros_data(new, trad)
                input_oh = np.tile(one_hot,(input_t.shape[0],1))
                input_t = np.concatenate((input_t, input_oh), axis=1)
                if name == self.test_name:
                    self.test_in = input_t
                    self.test_tar = target_t
                elif cnt == 0:
                    self.train_in = input_t
                    self.train_tar = target_t
                    cnt +=1
                else :
                    self.train_in = np.concatenate((self.train_in, input_t), axis=0)
                    self.train_tar = np.concatenate((self.train_tar, target_t), axis=0)
                    cnt += 1





    def rf_cv(self, C,epsilon,gamma):
        start = time.time()
        val = train_svr(self.train_in,self.train_tar,self.test_in,self.test_tar,C,epsilon,gamma)
        end = time.time()
        t = end-start
        print('time',t,'s')
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
            {'C': (1, 15),
             'epsilon': (0, 5),
             'gamma': (0.1, 100)}
        )
        rf_bo.maximize()
        lst = []
        for i in rf_bo.res:
            lst.append(i['target'])
        idx = lst.index(max(lst))
        best = rf_bo.res[idx]
        print(best)
        self.param = best['params']
        self.train_best()

    def train_best(self, path='./experiments/'):
        start = time.time()
        C = self.param['C']
        epsilon = self.param['epsilon']
        gamma = self.param['gamma']
        # C ,epsilon,gamma = self.param
        model = svm.SVR(degree=3, gamma=gamma, coef0=0.0, tol=0.005, C=C, epsilon=epsilon, shrinking=True,
                        cache_size=200, verbose=False, max_iter=- 1)
        name = str(C) + '.pkl'
        with open(path + name, "r") as f:
            model = pickle.load(f)


        model.fit(self.train_in, self.train_tar)
        pre = model.predict(self.test_in)
        metrics = [np_mae, np_rmse, np_mape, np_mmae]

        print("metrics1:\t[mae | rmse | mape | mmae]")
        print("results1:\t", [np.round(f(pre, self.test_tar), 3) for f in metrics])
        end = time.time()
        plt.title(self.type)
        plt.plot(self.test_in[:, 0], pre, label="pre")
        plt.plot(self.test_in[:, 0], self.test_tar, label="gt")
        # plt.plot(test[:, 0], pre2, label="pre2")
        # plt.plot(test[:, 0], pre3, label="pre3")
        # plt.plot(test[:, 0], pre4, label="pre4")
        # plt.plot(train[:, 0], train[:, 1]-train[:, 2], label="sub")
        plt.legend()
        plt.show()




if __name__ == "__main__":
    bys = Bayes(test_name = "13V",k=5)
    bys.opt()
>>>>>>> c78c412de2900af3ebb3032bd4f1c4e97c112707
