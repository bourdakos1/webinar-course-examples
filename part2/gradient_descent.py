from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

LEARNING_RATE = 0.0001
EPOCHS = 20000
LOG_EVERY_N_EPOCHS = 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1], use_bias=True))

weights = []
bias = []

class PrintWeights(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        weights.append(self.model.get_weights()[0][0][0])
        bias.append(self.model.get_weights()[1][0])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % LOG_EVERY_N_EPOCHS == 0):
            weights.append(self.model.get_weights()[0][0][0])
            bias.append(self.model.get_weights()[1][0])
            

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=LEARNING_RATE))

x_train = [-10.0, 15.0, 30.0]
y_train = [ 14.0, 59.0, 86.0]

# 5000 epochs to train to completion.
# F = 1.8 * C + 32
model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[PrintWeights()])

weights = np.array(weights)
bias = np.array(bias)

ax = plt.axes(projection='3d')

_w1 = np.linspace(-3, 6.6, 1000)
_b1 = np.linspace(-30, 94, 1000)
w1, b1 = np.meshgrid(_w1, _b1)

x1 = -10.0
target1 = 14.0

x2 = 15.0
target2 = 59.0

x3 = 30.0
target3 = 86.0

y1 = x1 * w1 + b1
cost1 = (target1 - y1) ** 2

y2 = x2 * w1 + b1
cost2 = (target2 - y2) ** 2

y3 = x3 * w1 + b1
cost3 = (target3 - y3) ** 2

# Gradient descent.
y1g = x1 * weights + bias
cost1g = (target1 - y1g) ** 2

y2g = x2 * weights + bias
cost2g = (target2 - y2g) ** 2

y3g = x3 * weights + bias
cost3g = (target3 - y3g) ** 2

ax.scatter(weights, bias, (cost1g + cost2g + cost3g) / 3, c='red')
ax.scatter(1.8, 32, c='green')

# ax.plot_surface(w1, b1, cost1, cmap="plasma")
ax.plot_surface(w1, b1, (cost1 + cost2 + cost3) / 3, cmap="plasma")

plt.show()