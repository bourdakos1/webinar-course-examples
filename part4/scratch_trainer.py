import random 
import math

def relu(x):
    return max(0, x)


def softmax(x, layer_values):
    top = math.exp(x)
    bottom = 0
    for v in layer_values:
        bottom = bottom + math.exp(v)
    return top / bottom



class Model:
    weights = []
    model_structure = []


    def add(self, units, input_shape=None, activation=None):
        self.model_structure.append((units, input_shape, activation))


    def compile(self):
        for (i, next_layer) in enumerate(self.model_structure):
            next_units, input_shape, _ = next_layer
            if i > 0:
                input_shape, _, _ = self.model_structure[i - 1]
 
            # glorot_uniform
            fan_in = input_shape
            fan_out = next_units
            limit = math.sqrt(6 / (fan_in + fan_out))

            weights = [[random.uniform(-limit, limit) for _ in range(0, next_units)] for _ in range(0, input_shape)]
            self.weights.append(weights)

            bias = [0.0 for _ in range(0, next_units)]
            self.weights.append(bias)


    def calc_layer(self, input_values, next_layer, weights, biases):
        if type(input_values) != list:
            input_values = [input_values]

        units, _, activation = next_layer
        next_layer_values = [0.0 for _ in range(0, units)]

        for (i, x) in enumerate(input_values):
            for (j, _) in enumerate(next_layer_values):
                next_layer_values[j] = next_layer_values[j] + x * weights[i][j]

        for (j, _) in enumerate(next_layer_values):
            next_layer_values[j] = next_layer_values[j] + 1 * biases[j]

        intermediate = [x for x in next_layer_values]

        if activation == 'relu':
            for (j, _) in enumerate(next_layer_values):
                next_layer_values[j] = relu(next_layer_values[j])
        elif activation == 'softmax':
            for (j, _) in enumerate(next_layer_values):
                next_layer_values[j] = softmax(next_layer_values[j], next_layer_values)

        activated = [x for x in next_layer_values]
        
        return (intermediate, activated)


    def calculate_feed_forward(self, xs):
        computed_graph = []
        activated = xs
        for (i, _) in enumerate(self.model_structure):
            (intermediate, activated) = self.calc_layer(activated, self.model_structure[i], self.weights[2 * i], self.weights[2 * i + 1])
            computed_graph.append(intermediate)
            computed_graph.append(activated)

        return computed_graph

    
    def predict(self, x):
        computed_graph = self.calculate_feed_forward(x)
        return computed_graph[len(computed_graph) - 1]

    
    def calculate_derivatives(self, xs, targets, computed_graph):
        w3 = self.weights[2][0][0]
        w4 = self.weights[2][0][1]
        w5 = self.weights[2][1][0]
        w6 = self.weights[2][1][1]

        _H1 = computed_graph[0][0]
        _H2 = computed_graph[0][1]
        H1 = computed_graph[1][0]
        H2 = computed_graph[1][1]
        _N = computed_graph[2][0]
        _P = computed_graph[2][1]
        N = computed_graph[3][0]
        P = computed_graph[3][1]

        EN_N = -targets[0] / N

        N_n = N * (1 - N)
        n_H1 = w3

        H1_h1 = 0
        if _H1 > 0:
            H1_h1 = 1

        h1_w1 = xs

        N_p = -N * P
        p_H1 = w4

        n_H2 = w5

        H2_h2 = 0
        if _H2 > 0:
            H2_h2 = 1

        p_H2 = w6

        h2_w2 = xs

        EP_P = -targets[1] / P

        P_n = -N * P
        P_p = P * (1 - P)
        
        n_w3 = H1
        p_w4 = H1
        n_w5 = H2
        p_w6 = H2

        n_b3 = 1
        p_b4 = 1
        h1_b1 = 1
        h2_b2 = 1

        dw1 = EN_N * (N_n * n_H1 * H1_h1 * h1_w1 + N_p * p_H1 * H1_h1 * h1_w1) + EP_P * (P_n * n_H1 * H1_h1 * h1_w1 + P_p * p_H1 * H1_h1 * h1_w1)
        dw2 = EN_N * (N_n * n_H2 * H2_h2 * h2_w2 + N_p * p_H2 * H2_h2 * h2_w2) + EP_P * (P_n * n_H2 * H2_h2 * h2_w2 + P_p * p_H2 * H2_h2 * h2_w2)
        dw3 = EN_N * N_n * n_w3 + EP_P * P_n * n_w3
        dw4 = EN_N * N_p * p_w4 + EP_P * P_p * p_w4
        dw5 = EN_N * N_n * n_w5 + EP_P * P_n * n_w5
        dw6 = EN_N * N_p * p_w6 + EP_P * P_p * p_w6

        db1 = EN_N * (N_n * n_H1 * H1_h1 * h1_b1 + N_p * p_H1 * H1_h1 * h1_b1) + EP_P * (P_n * n_H1 * H1_h1 * h1_b1 + P_p * p_H1 * H1_h1 * h1_b1)
        db2 = EN_N * (N_n * n_H2 * H2_h2 * h2_b2 + N_p * p_H2 * H2_h2 * h2_b2) + EP_P * (P_n * n_H2 * H2_h2 * h2_b2 + P_p * p_H2 * H2_h2 * h2_b2)
        db3 = EN_N * N_n * n_b3 + EP_P * P_n * n_b3
        db4 = EN_N * N_p * p_b4 + EP_P * P_p * p_b4

        return (dw1,dw2,dw3,dw4,dw5,dw6, db1,db2,db3,db4)


    def fit(self, x_train, y_train, epochs=5000, lr=0.01):
        for _ in range(0, 5000):
            sum_dw1 = 0
            sum_dw2 = 0
            sum_dw3 = 0
            sum_dw4 = 0
            sum_dw5 = 0
            sum_dw6 = 0

            sum_db1 = 0
            sum_db2 = 0
            sum_db3 = 0
            sum_db4 = 0

            for example in zip(x_train, y_train):
                xs, targets = example
                computed_graph = self.calculate_feed_forward(xs)
                (dw1,dw2,dw3,dw4,dw5,dw6, db1,db2,db3,db4) = self.calculate_derivatives(xs, targets, computed_graph)
                sum_dw1 = sum_dw1 + dw1
                sum_dw2 = sum_dw2 + dw2
                sum_dw3 = sum_dw3 + dw3
                sum_dw4 = sum_dw4 + dw4
                sum_dw5 = sum_dw5 + dw5
                sum_dw6 = sum_dw6 + dw6

                sum_db1 = sum_db1 + db1
                sum_db2 = sum_db2 + db2
                sum_db3 = sum_db3 + db3
                sum_db4 = sum_db4 + db4
            
            self.weights[0][0][0] = self.weights[0][0][0] - lr * sum_dw1 / len(x_train)
            self.weights[0][0][1] = self.weights[0][0][1] - lr * sum_dw2 / len(x_train)
            self.weights[2][0][0] = self.weights[2][0][0] - lr * sum_dw3 / len(x_train)
            self.weights[2][0][1] = self.weights[2][0][1] - lr * sum_dw4 / len(x_train)
            self.weights[2][1][0] = self.weights[2][1][0] - lr * sum_dw5 / len(x_train)
            self.weights[2][1][1] = self.weights[2][1][1] - lr * sum_dw6 / len(x_train)

            self.weights[1][0] = self.weights[1][0] - lr * sum_db1 / len(x_train)
            self.weights[1][1] = self.weights[1][1] - lr * sum_db2 / len(x_train)
            self.weights[3][0] = self.weights[3][0] - lr * sum_db3 / len(x_train)
            self.weights[3][1] = self.weights[3][1] - lr * sum_db4 / len(x_train)
            
    

x_train = [-5, 3, 7, -1]
y_train = [[1, 0], [0, 1], [0, 1], [1, 0]]

model = Model()
model.add(units=2, input_shape=1, activation='relu')
model.add(units=2, activation='softmax')
model.compile()

model.fit(x_train, y_train, epochs=5000, lr=0.1)

print('{:f}, {:f}'.format(*model.predict(-8)))
print('{:f}, {:f}'.format(*model.predict(1)))
print('{:f}, {:f}'.format(*model.predict(0)))
print('{:f}, {:f}'.format(*model.predict(9)))
print('{:f}, {:f}'.format(*model.predict(10)))
print('{:f}, {:f}'.format(*model.predict(-3)))

# TODO: we can get the following errors:
# OverflowError: math range error -- top = math.exp(x)
# ZeroDivisionError: float division by zero