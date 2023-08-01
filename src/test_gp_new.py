import os
import os.path as osp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import pdb

model = "../output/type2_0731/model/epoch10000"
save_dir = "./res"

expand_num = 5
from utils.mideadata import MideaData
from utils.evaluate import np_mae, np_mape, np_rmse
metrics = [np_mae, np_mape, np_rmse]

os.makedirs(save_dir, exist_ok=True)

m = tf.saved_model.load(model)

data = MideaData(cls=["13DKB"])
cls_ls = sorted(data.trad_data["13DKB"].keys(), key=lambda x: int(x[:-1]))
for cls in cls_ls:
    _, test_data = data.get_data(cls="13DKB", test_cls=cls)
    test_data = data.expand_data(test_data, expand_num)

    for te_name, te_data in test_data:
        te = te_data.copy()

        te[:, : 2 * expand_num + 1] /=  1e6
        
        pred_te_1 = te[te[:, expand_num] < 320]
        pred_te_2 = te[te[:, expand_num] >= 320]
        
        mean_1, _ = m.compiled_predict_f(pred_te_1[:, : 4 * expand_num + 2])
        mean_2, _ = m.compiled_predict_f(pred_te_2[:, : 4 * expand_num + 2])
        mean, var = m.compiled_predict_f(te[:, : 4 * expand_num + 2])
        mean, var = mean.numpy().reshape(-1), var.numpy().reshape(-1)

        _y_1 = mean_1.numpy().reshape(-1)
        _y_2 = mean_2.numpy().reshape(-1)

        print(f"{te_name=}")
        print("metrics:\t [mae | rmse | mape]")
        res_1 = [
            np.round(f(_y_1, pred_te_1[:, 4 * expand_num + 2]), 3) for f in metrics
        ]
        print(f"results: \t {res_1}")
        res_2 = [
            np.round(f(_y_2, pred_te_2[:, 4 * expand_num + 2]), 3) for f in metrics
        ]
        print(f"results: \t {res_2}")

        with open(os.path.join(save_dir, "res.txt"), "a") as f:
            f.write(f"{te_name}\n")
            f.write(f"{res_1}\n")
            f.write(f"{res_2}\n")
            f.write("\n")

        plt.figure(dpi=200)
        plt.clf()

        plt.plot(
            te[:, expand_num], te[:, 4 * expand_num + 2], label="ground truth", alpha=0.5
        )
        plt.plot(te[:, expand_num], mean, label="pred", alpha=0.5)
        plt.plot(te[:, expand_num], te[:, 3 * expand_num + 1], label="input", alpha=0.5)
        plt.fill_between(
            te[:, expand_num],
            np.ravel(mean + 2 * np.sqrt(var)),
            np.ravel(mean - 2 * np.sqrt(var)),
            alpha=0.3,
            color="red",
            label="95% Confidence Interval",
        )
        plt.title(f"{te_name}")
        plt.legend()
        plt.savefig(osp.join(save_dir, f"{te_name}_pred.png"))
