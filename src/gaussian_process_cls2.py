import os
import os.path as osp
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
from utils.mdlf import MideaDataLF
from utils.evaluate import np_mae, np_mape, np_rmse
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
@click.option("--test_cls", default="1H", help="iterations")
@click.option("--expand_num", default=5, help="expand number")
@click.option("--iterations", default=50000, help="iterations")
@click.option("--induce_num", default=1000, help="inducing points number")
@click.option("--minibatch_size", default=2000, help="minibatch size")
@click.option("--lr", default=1e-2, help="learning rate")
@click.option("--test_interval", default=2000, help="test interval")
@click.option("--save_dir", default="test5w", help="output save directory")
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
):
    os.makedirs(osp.join("../output", save_dir), exist_ok=True)
    save_dir = osp.join("../output", save_dir)

    metrics = [np_mae, np_rmse, np_mape]

    init_logging(save_dir)

    logging.info(
        f"{cls=}, {test_num=}, {test_cls=}, {expand_num=}, {induce_num=}, {iterations=}, {minibatch_size=}, {lr=}, {save_dir=}"
    )

    data = MideaDataLF()

    train_data, test_data = data.get_data(cls=cls, test_cls=test_cls)

    # [x * 101, trad * 101, new]

    train_data, test_data = data.expand_data(train_data, expand_num), data.expand_data(
        test_data, expand_num
    )

    tr = np.vstack([item[1] for item in train_data]).copy()

    # tr[:, : 2 * expand_num + 1] /= 1e6
    tr[:, : 4 * expand_num + 2] = (tr[:, : 4 * expand_num + 2] - tr[:, : 4 * expand_num + 2].min(axis=0)) / (tr[:, : 4 * expand_num + 2].max(axis=0) - tr[:, : 4 * expand_num + 2].min(axis=0))

    perm = np.random.permutation(len(tr))

    k = gpflow.kernels.Matern52(lengthscales=[100] * (4 * expand_num + 2), variance=1e2) # + gpflow.kernels.White(variance=1e2)

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

    for step in range(iterations):
        optimization_step()
        elbo = -training_loss().numpy()
        if step % 100 == 0:
            logging.info(f"Epoch: {step}: {elbo:.2f}")
        if step % test_interval == 0 or (step + 1) == iterations:
            logging.info(f"{m.kernel.variables=}")
            logging.info("testing ...")
            os.makedirs(osp.join(save_dir, f"epoch{step}"), exist_ok=True)
            for te_name, te_data in test_data:
                te = te_data.copy()

                te[:, : 2 * expand_num + 1] /= 1e6
                
                pred_te_1 = te[te[:, expand_num] < 320]
                pred_te_2 = te[te[:, expand_num] >= 320]
                
                min_1, max_1 = pred_te_1[:, : 4 * expand_num + 2].min(axis=0), pred_te_1[:, : 4 * expand_num + 2].max(axis=0)
                min_2, max_2 = pred_te_2[:, : 4 * expand_num + 2].min(axis=0), pred_te_2[:, : 4 * expand_num + 2].max(axis=0)

                pred_te_1[:, : 4 * expand_num + 2] = (pred_te_1[:, : 4 * expand_num + 2] - min_1) / (max_1 - min_1)
                pred_te_2[:, : 4 * expand_num + 2] = (pred_te_2[:, : 4 * expand_num + 2] - min_2) / (max_2 - min_2)
                
                mean_1, _ = m.predict_f(pred_te_1[:, : 4 * expand_num + 2])
                mean_2, _ = m.predict_f(pred_te_2[:, : 4 * expand_num + 2])

                _y_1 = mean_1.numpy().reshape(-1)
                _y_2 = mean_2.numpy().reshape(-1)

                logging.info(f"{te_name=}")
                logging.info("metrics:\t [mae | rmse | mape]")
                res = [
                    np.round(f(_y_1, pred_te_1[:, 4 * expand_num + 2]), 3) for f in metrics
                ]
                logging.info(f"(<30%) results: \t {res}")
                res = [
                    np.round(f(_y_2, pred_te_2[:, 4 * expand_num + 2]), 3) for f in metrics
                ]
                logging.info(f"(>30%) results: \t {res}")

                plt.figure()
                plt.clf()

                plt.plot(
                    te[:, expand_num], te[:, 4 * expand_num + 2], label="ground truth"
                )
                plt.plot(pred_te_1[:, expand_num] * (max_1[expand_num] - min_1[expand_num]) + min_1[expand_num], _y_1, label="pred")
                plt.legend()
                plt.savefig(osp.join(save_dir, f"epoch{step}", f"<30_{te_name}_pred.png"))

                plt.figure()
                plt.clf()

                plt.plot(
                    te[:, expand_num], te[:, 4 * expand_num + 2], label="ground truth"
                )
                plt.plot(pred_te_2[:, expand_num] * (max_2[expand_num] - min_2[expand_num]) + min_2[expand_num], _y_2, label="pred")
                plt.legend()
                plt.savefig(osp.join(save_dir, f"epoch{step}", f">30_{te_name}_pred.png"))


                # plt.figure()
                # plt.clf()

                # plt.plot(
                #     pred_te_1[:, expand_num],
                #     pred_te_1[:, 4 * expand_num + 2] - _y_1,
                #     label="error",
                # )
                # plt.legend()
                # plt.savefig(osp.join(save_dir, f"epoch{step}", f"<30_{te_name}_error.png"))
                
                # plt.figure()
                # plt.clf()

                # plt.plot(
                #     pred_te_2[:, expand_num],
                #     pred_te_2[:, 4 * expand_num + 2] - _y_2,
                #     label="error",
                # )
                # plt.legend()
                # plt.savefig(osp.join(save_dir, f"epoch{step}", f">30_{te_name}_error.png"))

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
                os.makedirs(osp.join(save_dir, f"epoch{step}", "model"), exist_ok=True)
                tf.saved_model.save(m, osp.join(save_dir, f"epoch{step}", "model"))


if __name__ == "__main__":
    GP()
