from scratch import Model 
    
# Our training data:
x_train = [-5, 3, 7, -1]
y_train = [[1, 0], [0, 1], [0, 1], [1, 0]]

# Initialize our model.
model = Model()

# Add a hidden layer with 2 nodes. Uses ReLu (Rectified Linear Units) as our 
# non-linearity.
model.add(units=2, input_shape=1, activation='relu') 

# Add our output with 2 nodes. It's activation is softmax, which is normally
# used for the last layer of classification networks.
# It normalizes the outputs into a probability distribution:
# 7 -> 0.87370430995
# 2 -> 0.00588697333
# 1 -> 0.00216569646
# 5 -> 0.11824302025
#   == 1.0
model.add(units=2, activation='softmax')

# Puts our network together. Initializes all the weights to random and biases to
# zero.
model.compile()

# Start training. Loop through each training example 5000 times. Uses a learning
# rate of 0.1.
model.fit(x_train, y_train, epochs=5000, lr=0.1)

# Print our results:
print('{:f}, {:f}'.format(*model.predict(-8)))
print('{:f}, {:f}'.format(*model.predict(1)))
print('{:f}, {:f}'.format(*model.predict(0)))
print('{:f}, {:f}'.format(*model.predict(9)))
print('{:f}, {:f}'.format(*model.predict(10)))
print('{:f}, {:f}'.format(*model.predict(-3)))