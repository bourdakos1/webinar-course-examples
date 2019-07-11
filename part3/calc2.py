import random 
import math

x = [-10.0, 15.0, 30.0]
target = [ 14.0, 59.0, 86.0]

# glorot_uniform
fan_in = 1 # the number of input units in the weight tensor
fan_out = 1 # the number of output units in the weight tensor
limit = math.sqrt(6 / (fan_in + fan_out))

W1 = random.uniform(-limit, limit) # ~ [-1.73, 1.73]
B1 = 0.0
W2 = random.uniform(-limit, limit)
B2 = 0.0

def feed_forward(x):
    H = x * W1 + B1
    y = H * W2 + B2
    return (H, y)

def calc_derivatives(x, target):
    (H, y) = feed_forward(x)

    dW2 = 0.0001 * -2 * (target - y) * H
    dB2 = 0.0001 * -2 * (target - y) * 1
    dW1 = 0.0001 * -2 * (target - y) * W2 * x
    dB1 = 0.0001 * -2 * (target - y) * W2 * 1

    return (dW2, dB2, dW1, dB1)

for step in range(0, 5000):
    sum_of_W1_derivatives = 0
    sum_of_B1_derivatives = 0
    sum_of_W2_derivatives = 0 
    sum_of_B2_derivatives = 0
    for sample in zip(x, target):
        (x_sample, target_sample) = sample
        (dW2, dB2, dW1, dB1) = calc_derivatives(x_sample, target_sample)
        sum_of_W1_derivatives = sum_of_W1_derivatives + dW1
        sum_of_B1_derivatives = sum_of_B1_derivatives + dB1
        sum_of_W2_derivatives = sum_of_W2_derivatives + dW2
        sum_of_B2_derivatives = sum_of_B2_derivatives + dB2

    W1 = W1 - sum_of_W1_derivatives / len(x)
    B1 = B1 - sum_of_B1_derivatives / len(x)
    W2 = W2 - sum_of_W2_derivatives / len(x)
    B2 = B2 - sum_of_B2_derivatives / len(x)


(_, answer) = feed_forward(22)
print(answer)
