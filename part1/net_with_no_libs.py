# Inputs
input0 = 10
input1 = 99

# Weights for addition
weights0 = [
    [ 1.0311713 ,  0.7898175 , -0.03300031,  0.03901571],
    [ 0.83759695, -0.44899732, -0.9797497 ,  0.33351913]
]
bias0 = [-0.10651446, 0.04045046, 0.1297183, -0.24981876]

weights1 = [
    [ 0.8379176 ],
    [ 0.15625027],
    [-0.3741129 ],
    [ 0.00534237]
]
bias1 = [0.13279381]

# Intermediate 
h0 = (input0 * weights0[0][0]) + (input1 * weights0[1][0]) + bias0[0]
h1 = (input0 * weights0[0][1]) + (input1 * weights0[1][1]) + bias0[1]
h2 = (input0 * weights0[0][2]) + (input1 * weights0[1][2]) + bias0[2]
h3 = (input0 * weights0[0][3]) + (input1 * weights0[1][3]) + bias0[3]

# Results
output = (h0 * weights1[0][0]) + (h1 * weights1[1][0]) + (h2 * weights1[2][0]) + (h3 * weights1[3][0]) + bias1[0]

print(output)