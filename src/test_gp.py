import os
import os.path as osp
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import pdb

res_dir = "../output/gp_0602"
last_epoch = 19999
expand_num = 50
from utils.mideadata import MideaData
from utils.evaluate import np_mae, np_mape, np_rmse
metrics = [np_mae, np_mape, np_rmse]

for cls in sorted(os.listdir(res_dir)):
    cls_model_res_dir = osp.join(res_dir, cls, f"epoch{last_epoch}", "model")
    m = tf.saved_model.load(cls_model_res_dir)
    
    data = MideaData()
    _, test_data = data.get_data(cls="13DKB", test_cls=cls)
    test_data = data.expand_data(test_data, expand_num)

    for te_name, te_data in test_data:
        te = te_data.copy()

        te[:, : 2 * expand_num + 1] /= 1e6
        pred_te_1 = te[te[:, expand_num] < 320]
        pred_te_2 = te[te[:, expand_num] >= 320]
        
        mean_1, _ = m.compiled_predict_f(pred_te_1[:, : 4 * expand_num + 2])
        mean_2, _ = m.compiled_predict_f(pred_te_2[:, : 4 * expand_num + 2])

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
        # plt.figure()
        # plt.clf()

        # plt.plot(
        #     te[:, expand_num], te[:, 4 * expand_num + 2], label="ground truth"
        # )
        # plt.plot(pred_te[:, expand_num], _y, label="pred")
        # plt.legend()
        # plt.savefig(osp.join(save_dir, f"epoch{step}", f"{te_name}_pred.png"))

        # plt.figure()
        # plt.clf()

        # plt.plot(
        #     pred_te[:, expand_num],
        #     pred_te[:, 4 * expand_num + 2] - _y,
        #     label="error",
        # )
        # plt.legend()
        # plt.savefig(osp.join(save_dir, f"epoch{step}", f"{te_name}_error.png"))