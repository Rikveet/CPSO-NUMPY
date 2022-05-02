from random import uniform, shuffle

import numpy
import numpy as np


class Network:
    def __init__(self, nodes_config, data):
        np.random.seed(10)
        self.train_output_data = None
        self.train_input_data = None
        input_layer_nodes, hidden_layer_nodes, num_hidden_layers, output_layer_nodes = nodes_config
        self.out_layer = output_layer_nodes
        max_weight, min_weight = (-1, 1)
        self.data = data

        # create weights and biases
        self.network_weights, self.network_biases = [], []

        # input layer
        self.network_weights \
            .append(np.random.uniform(low=min_weight, high=max_weight, size=(input_layer_nodes, hidden_layer_nodes))
                    .astype('float32'))

        # hidden layer
        self.network_biases.append(np.full((1, hidden_layer_nodes), 0.1).astype('float32'))
        for _ in range(num_hidden_layers - 1):
            self.network_weights \
                .append(np.random.uniform(low=min_weight, high=max_weight,
                                          size=(hidden_layer_nodes, hidden_layer_nodes)).astype('float32'))
            self.network_biases.append(np.full((1, hidden_layer_nodes), 0.1).astype('float32'))

        # output layer
        self.network_weights \
            .append(np.random.uniform(low=min_weight, high=max_weight, size=(hidden_layer_nodes, output_layer_nodes))
                    .astype('float32'))
        self.network_biases.append(np.full((1, output_layer_nodes), 0.1).astype('float32'))

        np.set_printoptions(precision=2)
        print("Network initialized")

    def shuffle_data(self):
        shuffle(self.data)

    def create_batch(self):
        self.shuffle_data()
        input_data = []
        output_data = []
        for i in range(100):
            inpt, outpt = self.data[i]
            input_data.append(inpt)
            output_data.append(outpt)
        self.train_input_data = numpy.array(input_data).astype('float32')
        self.train_output_data = numpy.array(output_data).astype('float32')

    def get_weight_decay(self):
        wd = 0
        for layer in self.network_weights:
            wd += np.sum(np.square(layer))
        return wd

    def feed_forward(self, show=False):
        value = 0
        value_layer = self.train_input_data
        for i in range(len(self.network_weights)):
            value_layer = np.maximum(np.dot(value_layer, self.network_weights[i]) + self.network_biases[i], 0)
        value += np.sum(np.square(value_layer - self.train_output_data))
        if show:
            for i in range(100):
                print(self.train_input_data[i], "::", self.train_output_data[i], "::", value_layer[i])  # )
        return value / (self.out_layer * 100)

    def update_weights(self, params, value):
        layer, node, out_node = params
        if out_node == -1:  # if it is a bias
            self.network_biases[layer - 1][0, node] = value
        else:
            self.network_weights[layer][node, out_node] = value

    def test(self, params, value):
        layer, node, out_node = params
        if out_node == -1:
            current_value = self.network_biases[layer - 1][0, node]
            self.network_biases[layer - 1][0, node] = value
            test_value = self.feed_forward()
            self.network_biases[layer - 1][0, node] = current_value
        else:
            current_value = self.network_weights[layer][node, out_node]
            self.network_weights[layer][node, out_node] = value
            test_value = self.feed_forward()
            self.network_weights[layer][node, out_node] = current_value
        return test_value
