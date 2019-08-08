import random 
import math

# TODO: we can get the following errors:
# OverflowError: math range error -- math.exp(x)
# ZeroDivisionError: float division by zero

def relu(x):
    return max(0, x)

def softmax(x, layer_values):
    # e^layer[x] / (e^layer[0] + e^layer[1] + ... + e^layer[n])
    top = math.exp(x)
    bottom = 0
    for v in layer_values:
        bottom = bottom + math.exp(v)
    return top / bottom


class Model:
    weights = []
    model_structure = []


    def add(self, units, input_shape=None, activation=None):
        # Simply add each layer to an array so we can keep track of our graph's
        # (aka model's) structure.
        self.model_structure.append((units, input_shape, activation))


    def compile(self):
        # Loop through each layer in our model. We call it `next_layer`, because
        # the input layer is our first layer and the first item in our 
        # `model_structure` is really the next layer.
        for (i, next_layer) in enumerate(self.model_structure):
            # Get the input shape from the layer. This will be ignored if it's
            # not the first layer in our `model_structure`
            next_units, input_shape, _ = next_layer

            # If this isn't the first layer, replace `input_shape` with the
            # previous layer's size.
            if i > 0:
                # This replaces `input_shape` with `units` of the previous layer.
                input_shape, _, _ = self.model_structure[i - 1]
 
            # `glorot_uniform` weight initialization. Why do we do this? 
            # Check out this research paper:
            # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            # TLDR: because it helps the model learn faster.
            fan_in = input_shape
            fan_out = next_units
            limit = math.sqrt(6 / (fan_in + fan_out))

            # Set all the weights in the layer to random.
            weights = [[random.uniform(-limit, limit) for _ in range(0, next_units)] for _ in range(0, input_shape)]
            self.weights.append(weights)

            # Set all the biases in the layer to zero.
            bias = [0.0 for _ in range(0, next_units)]
            self.weights.append(bias)


    def fit(self, x_train, y_train, epochs=1, lr=0.01):
        # Loop for `epochs` i.e. If epochs=5000, loop 5000 times.
        for _ in range(0, epochs):
            # Ideally we would store these in a list, like the weights list.
            # What we have now isn't flexible, but a list would be.
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

            # Loop through each training example.
            for example in zip(x_train, y_train):
                xs, targets = example

                # Pass the inputs (Xs) through the graph keep all the 
                # intermediate values.
                computed_graph = self.calculate_feed_forward(xs)

                # We use the intermediate values (computed_graph), targets, and 
                # inputs to compute the derivatives for each weight and bias.
                (dw1,dw2,dw3,dw4,dw5,dw6, db1,db2,db3,db4) = self.calculate_derivatives(xs, targets, computed_graph)

                # Add each derivative to the sum of derivatives for the current 
                # epoch.
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
            
            # At the end of the epoch, subtract the sum of the weight's/bias's
            # derivative (multiplied by the learning rate) from the weight/bias.
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


    def calculate_feed_forward(self, xs):
        computed_graph = []
        activated = xs
        # Loop through each layer, passing in the previous layer's activations,
        # and storing the intermediate and activation.
        # The intermediate is the value of the node before going through the 
        # activation function.
        for (i, _) in enumerate(self.model_structure):
            (intermediate, activated) = self.calc_layer(activated, self.model_structure[i], self.weights[2 * i], self.weights[2 * i + 1])
            computed_graph.append(intermediate)
            computed_graph.append(activated)

        return computed_graph


    def calc_layer(self, input_values, next_layer, weights, biases):
        # Make sure `input_values` is a list, because it could be a single value.
        if type(input_values) != list:
            # If it's not a list, just put it into a list.
            input_values = [input_values]

        # Find out how many units are in the next layer and what activation 
        # function it has.
        units, _, activation = next_layer

        # Initialize a list to store the results.
        next_layer_values = [0.0 for _ in range(0, units)]

        # Loop through each input value and multiply it by each of it's weights,
        # adding the result to the value in `next_layer_values` list.
        # This could be achieved more efficiently with `numpy` and matrix 
        # multiplication.
        # 
        # Example loop:
        # input_values = [1, 2]
        # weights = [[1, 2], [3, 4]]
        # next_values = [0, 0]
        # next_values[0] = next_values[0] + input_values[0] * weights[0][0]
        #      1         =      0         +        1        *       1
        # next_values[1] = next_values[1] + input_values[0] * weights[0][1]
        #      2         =      0         +        1        *       2
        # next_values[0] = next_values[0] + input_values[1] * weights[1][0]
        #      7         =      1         +        2        *       3
        # next_values[1] = next_values[1] + input_values[1] * weights[1][1]
        #     10         =      2         +        2        *       4
        for (i, x) in enumerate(input_values):
            for (j, weight) in enumerate(weights[i]):
                next_layer_values[j] = next_layer_values[j] + x * weight

        # Do the same with biases. Example:
        # biases = [5, 6]
        # next_values = [7, 10]
        # next_values[0] = next_values[0] + biases[0]
        #     12         =      7         +    5
        # next_values[1] = next_values[1] + biases[1]
        #     16         =     10         +    6
        for (j, bias) in enumerate(biases):
            next_layer_values[j] = next_layer_values[j] + 1 * bias

        # Make a copy of the `next_layer_values` before we calculate the 
        # activations.
        intermediate = [x for x in next_layer_values]

        # Loop through each value and apply the activation specified.
        if activation == 'relu':
            for (j, _) in enumerate(next_layer_values):
                next_layer_values[j] = relu(next_layer_values[j])
        elif activation == 'softmax':
            for (j, _) in enumerate(next_layer_values):
                next_layer_values[j] = softmax(next_layer_values[j], next_layer_values)

        # Make a copy of the `next_layer_values` (not really necessary to make a
        # deep copy, but I like the symmetry)
        activated = [x for x in next_layer_values]
        
        # return the copies of the intermediates and activations.
        return (intermediate, activated)


    def calculate_derivatives(self, xs, targets, computed_graph):
        # Hard coded derivatives. We will need to redo this if we want to train
        # a network that is a different shape than our 1 -> 2 -> 2 model.
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


    def predict(self, x):
        # Pass the value through the graph.
        computed_graph = self.calculate_feed_forward(x)
        # return the last layer's values.
        return computed_graph[len(computed_graph) - 1]