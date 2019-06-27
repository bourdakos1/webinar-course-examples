import numpy as np

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1], use_bias=True))

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.1))

x_train = [-40.0, -10.0,  0.0,  8.0, 15.0, 25.0,  38.0,  42.0]
y_train = [-40.0,  14.0, 32.0, 46.4, 59.0, 77.0, 100.4, 107.6]

model.fit(x_train, y_train, epochs=1000)

print(model.get_weights())

# F = 1.8 * C + 32
TODAYS_TEMP = 22
print(model.predict([[TODAYS_TEMP]]))