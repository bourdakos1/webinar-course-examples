import numpy as np

import tensorflow as tf

# def function(x):
#     y = 32 + 1.8 * x
#     return y

def function(x):
    y = x ** 2
    return y

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_shape=[1]))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=1))

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

x_train = [-300, -100, -40, -10,  0,  8, 15, 22, 38, 100, 300]
y_train = np.array([function(x) for x in x_train])

model.fit(x_train, y_train, epochs=500)

import matplotlib.pyplot as plt
test_labels = np.transpose(np.arange(-300, 300))
test_predictions = model.predict(test_labels)
real_predictions = [function(x) for x in test_labels]
plt.plot(test_labels, test_predictions, marker=',')
plt.plot(test_labels, real_predictions, color='red', marker=',')
plt.show()