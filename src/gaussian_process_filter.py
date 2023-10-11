import os
import os.path as osp
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
from utils.mideadata import MideaData
from utils.mdlf import MideaDataLF
from utils.evaluate import np_mae, np_mape, np_rmse
from  utils.filter import filter_peak, expand_data
import logging
import click
import pdb


def init_logging(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=osp.join(save_dir, "log.log"),
    )
    logging.getLogger().addHandler(logging.StreamHandler())


@click.command()
@click.option("--cls", default="13DKB", help="class name")
@click.option("--test_num", default=1, help="test data number")
@click.option("--test_cls", default="2H", help="iterations")
@click.option("--expand_num", default=5, help="expand number")
@click.option("--iterations", default=10000, help="iterations")
@click.option("--induce_num", default=1500, help="inducing points number")
@click.option("--minibatch_size", default=2000, help="minibatch size")
@click.option("--lr", default=2e-3, help="learning rate")
@click.option("--test_interval", default=500, help="test interval")
@click.option("--save_dir", default="test_2H", help="output save directory")

@click.option("--filter_win", default=200, help="window size for filter", type=int)
@click.option("--peak_thresh", default=20, help="peak thresh", type=int)
@click.option("--peak_space", default=300, help="peak space", type=int)

def GP(
    cls,
    test_num,
    test_cls,
    expand_num,
    iterations,
    induce_num,
    minibatch_size,
    lr,
    test_interval,
    save_dir,
    filter_win,
    peak_thresh,
    peak_space
):
    os.makedirs(osp.join("../output", save_dir), exist_ok=True)
    save_dir = osp.join("../output", save_dir)
    # os.makedirs(osp.join(save_dir, "base/model"), exist_ok=True)
    # os.makedirs(osp.join(save_dir, "peak/model"), exist_ok=True)
    os.makedirs(osp.join(save_dir, "model"), exist_ok=True)
    
    metrics = [np_mae, np_rmse, np_mape]

    init_logging(save_dir)

    # logging.info(
    #     f"{cls=}, {test_num=}, {test_cls=}, {expand_num=}, {induce_num=}, {iterations=}, {minibatch_size=}, {lr=}, {save_dir=}"
    # )

    data = MideaData(cls=[cls])
    #
    # if cls == "13DKB":
    #     data.gen_mount_data(win=7, scale=2.)
    #     data.plot_data(name=osp.join(save_dir, "mount_data.png"))
    #
    # if cls == "13DKB":
    #     train_data, _ = data.get_data(cls=cls, test_cls=test_cls)
    #     test_data, _ = data.get_ori_data(cls=cls, test_cls=test_cls)
    # else:
    #     train_data, test_data = data.get_ori_data(cls=cls, test_cls=test_cls)

    if cls == "13DKB":
        filter_train_data, filter_test_data, peak_train_list, peak_test_list = filter_peak(
            cls = cls,
            test_cls = test_cls,
            filter_win = filter_win,
            peak_thresh = peak_thresh,
            peak_space = peak_space
        )
        
    train_data = data.expand_data(filter_train_data.copy(), expand_num)
    test_data = data.expand_data(filter_test_data.copy(), expand_num)
    
    #train base model   
    tr = np.vstack([item[1] for item in train_data]).copy()

    tr[:, : 2 * expand_num + 1] /= 1e6

    perm = np.random.permutation(len(tr))

    k = gpflow.kernels.Matern52(lengthscales=[50] * (4 * expand_num + 2), variance=1e2)

    M = induce_num

    Z = tr[perm[:M]][:, : 4 * expand_num + 2].copy()

    m = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(), Z, num_data=len(tr))

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tr[:, : 4 * expand_num + 2].reshape(-1, 4 * expand_num + 2),
                tr[:, 4 * expand_num + 2].reshape(-1, 1),
            )
        )
        .repeat()
        .shuffle(len(tr))
    )

    train_iter = iter(train_dataset.batch(minibatch_size))

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    training_loss = m.training_loss_closure(train_iter, compile=True)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, m.trainable_variables)

    for step in range(1, iterations + 1):
        optimization_step()
        elbo = -training_loss().numpy()
        if step % 100 == 0:
            logging.info(f"Epoch: {step}: {elbo:.2f}")
        if step % test_interval == 0 or step == iterations:
            # logging.info(f"{m.kernel.variables=}")
            logging.info("testing ...")
            os.makedirs(osp.join(save_dir, "base", f"epoch{step}"), exist_ok=True)
            for te_name, te_data in test_data:
                te = te_data.copy()

                te[:, : 2 * expand_num + 1] /= 1e6
                pred_te_1 = te[te[:, expand_num] < 320]
                pred_te_2 = te[te[:, expand_num] >= 320]
                
                mean_1, _ = m.predict_f(pred_te_1[:, : 4 * expand_num + 2])
                mean_2, _ = m.predict_f(pred_te_2[:, : 4 * expand_num + 2])
                mean, var = m.predict_f(te[:, : 4 * expand_num + 2])
                mean, var = mean.numpy().reshape(-1), var.numpy().reshape(-1)
                
                _y_1 = mean_1.numpy().reshape(-1)
                _y_2 = mean_2.numpy().reshape(-1)

                # logging.info(f"{te_name=}")
                logging.info("metrics:\t [mae | rmse | mape]")
                res = [
                    np.round(f(_y_1, pred_te_1[:, 4 * expand_num + 2]), 3) for f in metrics
                ]
                logging.info(f"(<30%) results: \t {res}")
                res = [
                    np.round(f(_y_2, pred_te_2[:, 4 * expand_num + 2]), 3) for f in metrics
                ]
                logging.info(f"(>30%) results: \t {res}")

                plt.figure(dpi=300)
                plt.clf()

                plt.plot(
                    te[:, expand_num], te[:, 4 * expand_num + 2], alpha=0.7, label="ground truth"
                )
                plt.plot(te[:, expand_num], mean, alpha=0.7, label="pred")
                
                plt.plot(te[:, expand_num], te[:, 3 * expand_num + 1], alpha=0.7, label="input")
                plt.fill_between(
                    te[:, expand_num],
                    np.ravel(mean + 2 * np.sqrt(var)),
                    np.ravel(mean - 2 * np.sqrt(var)),
                    alpha=0.3,
                    color="red",
                    label="95% CI",
                )
                plt.legend()
                plt.savefig(osp.join(save_dir, "base", f"epoch{step}", f"{te_name}_pred.png"))

            if step != 0:
                m.compiled_predict_f = tf.function(
                    lambda Xnew: m.predict_f(Xnew, full_cov=False),
                    input_signature=[
                        tf.TensorSpec(
                            shape=[None, 4 * expand_num + 2], dtype=tf.float64
                        )
                    ],
                )
                m.compiled_predict_y = tf.function(
                    lambda Xnew: m.predict_y(Xnew, full_cov=False),
                    input_signature=[
                        tf.TensorSpec(
                            shape=[None, 4 * expand_num + 2], dtype=tf.float64
                        )
                    ],
                )
                os.makedirs(osp.join(save_dir, "base", "model", f"epoch{step}"), exist_ok=True)
                tf.saved_model.save(m, osp.join(save_dir, "base", "model", f"epoch{step}"))           
              
    # train peak model
    
    model_base = tf.saved_model.load(osp.join(save_dir, "base", "model", "epoch5000"))
     
    
    peak_train_data = []

    for i in range(len(peak_train_list)):
        (name, peak_list) = peak_train_list[i]
        if len(peak_list) > 0:
            # predict base
            _, filter_data, _ = filter_train_data[i]

            filter_data[:, 0] /=1e6
            mean, _ = model_base.compiled_predict_f(expand_data(filter_data, 5)[:, :-1])
            pred_base = mean.numpy().reshape(-1,1)
            
            ori_data = filter_train_data[i][-1]
            train_data = []
            for peak_tuple in peak_list:
                idx, delta = peak_tuple
                idx = int(idx)
                delta1 = np.max(ori_data[idx-peak_space//2:idx+peak_space//2, 2]) - pred_base[idx]# array([1])
                train_data.append([ori_data[idx, 0], delta, delta1[0]])
            train_data = np.asarray(train_data)
            peak_train_data.append(
                (
                    name,
                    train_data
                )
            )
    
    peak_test_data = []
    for i in range(len(peak_test_list)):
        (name, peak_list) = peak_test_list[i]
        if len(peak_list) > 0:
            # predict base
            _, filter_data, _ = filter_test_data[i]
            filter_data[:, 0] /=1e6
            
            mean_base, var_base = model_base.compiled_predict_f(expand_data(filter_data, 5)[:, :-1])
            mean_base, var_base = mean_base.numpy().reshape(-1), var_base.numpy().reshape(-1)

            pred_base = mean.numpy().reshape(-1,1)
            _, filt_data, ori_data = filter_test_data[i]
            gt = filter_test_data[i][1][:, -1].copy()
            test_data = []
            for (idx,delta) in peak_list:
                idx = int(idx)
                delta1 = np.max(ori_data[idx-peak_space//2:idx+peak_space//2, 2]) - pred_base[idx] # array([1])
                test_data.append([idx, delta, delta1[0]])
                gt[idx] += delta1[0]
            
            test_data = np.asarray(test_data)
            print(test_data)
            peak_test_data.append(
                (
                    name,
                    test_data
                )
            )
    train_data = data.expand_data(peak_train_data.copy(), expand_num)
    test_data = data.expand_data(peak_test_data.copy(), expand_num)
        
    tr = np.vstack([item[1] for item in train_data]).copy()

    tr[:, : 2 * expand_num + 1] /= 1e6

    perm = np.random.permutation(len(tr))

    k = gpflow.kernels.Matern52(lengthscales=[50] * (4 * expand_num + 2), variance=1e2)

    M = induce_num

    Z = tr[perm[:M]][:, : 4 * expand_num + 2].copy()

    m = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(), Z, num_data=len(tr))

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tr[:, : 4 * expand_num + 2].reshape(-1, 4 * expand_num + 2),
                tr[:, 4 * expand_num + 2].reshape(-1, 1),
            )
        )
        .repeat()
        .shuffle(len(tr))
    )

    train_iter = iter(train_dataset.batch(minibatch_size))

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    training_loss = m.training_loss_closure(train_iter, compile=True)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, m.trainable_variables)

    for step in range(1, iterations + 1):
        optimization_step()
        elbo = -training_loss().numpy()
        if step % 100 == 0:
            logging.info(f"Epoch: {step}: {elbo:.2f}")
        if step % test_interval == 0 or step == iterations:
            # logging.info(f"{m.kernel.variables=}")
            logging.info("testing ...")
            os.makedirs(osp.join(save_dir, "peak", f"epoch{step}"), exist_ok=True)
            for te_name, te_data in test_data:
                te = te_data.copy()

                te[:, : 2 * expand_num + 1] /= 1e6

                mean, var = m.predict_f(te[:, : 4 * expand_num + 2])
                mean, var = mean.numpy().reshape(-1), var.numpy().reshape(-1)

                _y =  mean
                print(_y)

                for i in range(len(peak_test_list)):

                    (name, peak_list) = peak_test_list[i]
                    if len(peak_list) > 0:
                        # predict base
                        _, filter_data, _ = filter_test_data[i]
                        pre =  mean_base.copy()
                        pre_var = var_base.copy()

                        for i in range(len(peak_list)):
                            idx = int(peak_list[i][0])
                            #计算预测值
                            pre[idx] = mean_base[idx] + _y[i]
                            pre_var[idx] = var[i]
                        
                r = pre.shape[0] * 0.3
                r = int(r)
                _y_1 = pre[:r]
                _y_2 = pre[r:]
                # logging.info(f"{te_name=}")
                
                
                logging.info("metrics:\t [mae | rmse | mape]")
                res = [
                    np.round(f(_y_1, gt[:r]), 3) for f in metrics
                ]
                logging.info(f"(<30%) results: \t {res}")
                res = [
                    np.round(f(_y_2, gt[r:]), 3) for f in metrics
                ]
                logging.info(f"(>30%) results: \t {res}")

                plt.figure(dpi=300)
                plt.clf()

                plt.plot(
                    filt_data[:, 0], gt, alpha=0.7, label="ground truth"
                )
                plt.plot(filt_data[:, 0], pre, alpha=0.7, label="pred")
                
                plt.plot(filt_data[:, 0], filt_data[:, 1], alpha=0.7, label="input")
                plt.fill_between(
                    filt_data[:, 0],
                    np.ravel(pre + 2 * np.sqrt(pre_var)),
                    np.ravel(pre - 2 * np.sqrt(pre_var)),
                    alpha=0.3,
                    color="red",
                    label="95% CI",
                )
                plt.legend()
                plt.savefig(osp.join(save_dir, "peak", f"epoch{step}", f"{te_name}_pred.png"))

            if step != 0:
                m.compiled_predict_f = tf.function(
                    lambda Xnew: m.predict_f(Xnew, full_cov=False),
                    input_signature=[
                        tf.TensorSpec(
                            shape=[None, 4 * expand_num + 2], dtype=tf.float64
                        )
                    ],
                )
                m.compiled_predict_y = tf.function(
                    lambda Xnew: m.predict_y(Xnew, full_cov=False),
                    input_signature=[
                        tf.TensorSpec(
                            shape=[None, 4 * expand_num + 2], dtype=tf.float64
                        )
                    ],
                )
                os.makedirs(osp.join(save_dir, "peak", "model", f"epoch{step}"), exist_ok=True)
                tf.saved_model.save(m, osp.join(save_dir, "peak", "model", f"epoch{step}"))


if __name__ == "__main__":
    GP()
