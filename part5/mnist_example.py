import tensorflow as tf
import tensorflowjs as tfjs

# Keras comes with some standard datasets that we can easily import. We are 
# using MNIST here, which is a dataset of handwritten digits. The data is split 
# into training and test sets.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert the values to floats.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize pixel values to be between 0 and 1.
x_train /= 255
x_test /= 255

# Convert class vectors to one hot encoded binary class matrices.
# For example with 5 classes and 8 examples:
# [4, 1, 3, 2, 5, 1, 3, 2] -> goes to:
# [[0 0 0 1 0],
#  [1 0 0 0 0],
#  [0 0 1 0 0],
#  [0 1 0 0 0],
#  [0 0 0 0 1],
#  [1 0 0 0 0],
#  [0 0 1 0 0],
#  [0 1 0 0 0]]
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential()
# 28 pixels by 28 pixels 
# with color images we would need a 3rd channel with 3 values (Red, Green, Blue)
# e.g. input_shape=(28, 28, 3)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=15)

# Run the 1000 test examples and print the accuracy.
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model as a standard Keras model and as a TensorFlow.js model.
model.save('mnist_model.h5')
tfjs.converters.save_keras_model(model, 'public/mnist_model')