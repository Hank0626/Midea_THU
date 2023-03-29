import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt

from dataload import MideaData
from evaluate import np_mae, np_mape, np_rmse

data = MideaData()

train, test = data.get_data("data_chu_1a", 100)

perm = np.random.permutation(len(train))

k = gpflow.kernels.Matern52(lengthscales=[0.01, 0.01])

M = 100

Z = train[perm[:M]][:, [0, 2]].copy()

m = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(), Z, num_data=len(train))

minibatch_size = 100

train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (train[:, [0, 2]].reshape(-1, 2), train[:, 1].reshape(-1, 1))
    )
    .repeat()
    .shuffle(len(train))
)

train_iter = iter(train_dataset.batch(minibatch_size))

optimizer = tf.optimizers.Adam()
training_loss = m.training_loss_closure(train_iter, compile=True)

iterations = 10000


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, m.trainable_variables)


for step in range(iterations):
    optimization_step()
    if step % 500 == 0:
        elbo = -training_loss().numpy()
        print(step, elbo)

mean, var = m.predict_f(test[:, [0, 2]])

_y = mean.numpy().reshape(-1)

metrics = [np_mae, np_rmse, np_mape]

print("metrics:\t[mae | rmse | mape]")
print("results:\t", [np.round(f(_y, test[:, 1]), 3) for f in metrics])

plt.plot(test[:, 0], test[:, 1], label="ground truth")
plt.plot(test[:, 0], _y, label="pred")
plt.legend()
plt.show()

plt.plot(test[:, 0], test[:, 1] - _y, label="error")
plt.legend()
plt.show()
