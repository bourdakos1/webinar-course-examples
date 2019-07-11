x = [-10.0, 15.0, 30.0]
target = [ 14.0, 59.0, 86.0]

W1 = 0.87209165
B1 = 0.0
W2 = 1.0028847
B2 = 0.0

def calc_derivatives(x, target):
    H = x * W1 + B1
    y = H * W2 + B2

    dW2 = 0.0001 * -2 * (target - y) * H
    dB2 = 0.0001 * -2 * (target - y) * 1
    dW1 = 0.0001 * -2 * (target - y) * W2 * x
    dB1 = 0.0001 * -2 * (target - y) * W2 * 1

    return (dW2, dB2, dW1, dB1)

(dW2_0, dB2_0, dW1_0, dB1_0) = calc_derivatives(x[0], target[0])
(dW2_1, dB2_1, dW1_1, dB1_1) = calc_derivatives(x[1], target[1])
(dW2_2, dB2_2, dW1_2, dB1_2) = calc_derivatives(x[2], target[2])

nw1 = W1 - (dW1_0 + dW1_1 + dW1_2) / 3
nb1 = B1 - (dB1_0 + dB1_1 + dB1_2) / 3
nw2 = W2 - (dW2_0 + dW2_1 + dW2_2) / 3
nb2 = B2 - (dB2_0 + dB2_1 + dB2_2) / 3

# 1.0227654
# 0.00858394
# 1.1339082
# 0.00855925

print(nw1)
print(nb1)
print(nw2)
print(nb2)
