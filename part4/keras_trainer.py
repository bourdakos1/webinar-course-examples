import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=2, input_shape=[1], use_bias=True, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, use_bias=True, activation='softmax'))

class PrintWeights(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('Beggining to train with weights:')
        print(self.model.get_weights())
        print('')

    def on_epoch_end(self, epoch, logs=None):
        print('Weight update:')
        print(self.model.get_weights())
        print('')

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01))

x_train = [-5, 3, 7, -1]
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# # 5000 epochs to train to ~completion.
# model.fit(x_train, y_train, epochs=1, verbose=0, callbacks=[PrintWeights()])

np.set_printoptions(suppress=True)
model.fit(x_train, y_train, epochs=5000)
print(model.predict([-8, 1, 0, 9, 10, -3]))