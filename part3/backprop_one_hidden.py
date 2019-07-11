import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1], use_bias=True))
model.add(tf.keras.layers.Dense(units=1, use_bias=True))

class PrintWeights(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('Beggining to train with weights:')
        print(self.model.get_weights())
        print('')

    def on_epoch_end(self, epoch, logs=None):
        print('Weight update:')
        print(self.model.get_weights())
        print('')

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.0001))

x_train = [-10.0, 15.0, 30.0]
y_train = [ 14.0, 59.0, 86.0]

# 5000 epochs to train to completion.
# F = 1.8 * C + 32
model.fit(x_train, y_train, epochs=1, verbose=0, callbacks=[PrintWeights()])
# model.fit(x_train, y_train, epochs=5000)