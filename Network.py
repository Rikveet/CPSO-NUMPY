import numpy
import numpy as np


class Network:
    def __init__(self, random, nodes_config, lda, data):
        np.seterr(divide='ignore', invalid='ignore')
        self.random = random
        self.lda = lda
        np.random.seed(10)
        self.weight_decay = None
        self.train_input_data = None
        self.train_output_data = None
        self.test_input_data = None
        self.test_output_data = None
        self.batch_train_input_data = None
        self.batch_train_output_data = None
        input_layer_nodes, hidden_layer_nodes, num_hidden_layers, output_layer_nodes = nodes_config
        self.out_layer = output_layer_nodes
        max_weight, min_weight = (-1, 1)
        self.data = data
        self.load_all_data()
        # create weights and biases
        self.network_weights, self.network_biases = [], []

        # input layer weight and bias
        self.network_weights \
            .append(np.random.uniform(low=min_weight, high=max_weight, size=(input_layer_nodes, hidden_layer_nodes))
                    .astype('float32'))
        self.network_biases.append(np.full((1, hidden_layer_nodes), 0.1).astype('float32'))

        # hidden layer weight and bias
        self.network_biases.append(np.full((1, output_layer_nodes), 0.1).astype('float32'))
        for _ in range(num_hidden_layers - 1):
            self.network_weights \
                .append(np.random.uniform(low=min_weight, high=max_weight,
                                          size=(hidden_layer_nodes, hidden_layer_nodes)).astype('float32'))
            self.network_biases.append(np.full((1, hidden_layer_nodes), 0.1).astype('float32'))

        # output layer
        self.network_weights \
            .append(np.random.uniform(low=min_weight, high=max_weight, size=(hidden_layer_nodes, output_layer_nodes))
                    .astype('float32'))
        self.update_weight_decay()
        np.set_printoptions(precision=2)

    def shuffle_data(self):
        self.random.shuffle(self.data)

    def create_batch(self):
        self.shuffle_data()
        input_data = []
        output_data = []
        for i in range(100):  # 100
            inpt, outpt = self.data[i]
            input_data.append(inpt)
            output_data.append(outpt)
        self.batch_train_input_data = numpy.array(input_data).astype('float32')
        self.batch_train_output_data = numpy.array(output_data).astype('float32')

    def load_all_data(self):
        self.shuffle_data()
        train_input_data = []
        train_output_data = []
        test_input_data = []
        test_output_data = []
        train_limit = int(len(self.data) * 0.80)
        for i in range(train_limit):
            Input, Output = self.data[i]
            train_input_data.append(Input)
            train_output_data.append(Output)
        for i in range(train_limit, len(self.data)):
            Input, Output = self.data[i]
            test_input_data.append(Input)
            test_output_data.append(Output)

        self.train_input_data = numpy.array(train_input_data).astype('float32')
        self.train_output_data = numpy.array(train_output_data).astype('float32')
        self.test_input_data = numpy.array(test_input_data).astype('float32')
        self.test_output_data = numpy.array(test_output_data).astype('float32')

    def update_weight_decay(self):
        wd = 0
        for layer in self.network_weights:
            wd += np.sum(np.square(layer))
        for bias_layer in self.network_biases:
            wd += np.sum(np.square(bias_layer))
        self.weight_decay = (wd / (wd + 1)) * self.lda

    def feed_forward(self):
        value_layer = self.batch_train_input_data
        for i in range(len(self.network_weights)):  # batch
            value_layer = np.maximum(np.dot(value_layer, self.network_weights[i]) + self.network_biases[i], 0)
        avg = (np.sum(value_layer, axis=1)).reshape((len(value_layer), 1))
        avg[avg == 0] = 0.001
        value_layer = value_layer / avg
        value = np.sum(np.square(value_layer - self.batch_train_output_data))
        fitness = value / (self.out_layer * len(self.batch_train_input_data))
        return fitness + self.weight_decay

    def feed_forward_global(self, show=False):
        value_layer = self.train_input_data
        for i in range(len(self.network_weights)):  # whole
            value_layer = np.maximum(np.dot(value_layer, self.network_weights[i]) + self.network_biases[i], 0)
        avg = (np.sum(value_layer, axis=1)).reshape((len(value_layer), 1))
        avg[avg == 0] = 0.001
        value_layer = value_layer / avg
        value = np.sum(np.square(value_layer - self.train_output_data))
        global_train_fitness = value / (self.out_layer * len(self.train_input_data))
        if show:
            value_layer = np.round(value_layer, 2)
            for i in range(len(self.train_input_data)):
                print("{}..".format(str(self.train_input_data[i])[:4]), "::", self.train_output_data[i], "::",
                      value_layer[i])  # )
        return global_train_fitness + self.weight_decay

    def feed_forward_global_test(self, show=False):
        value_layer = self.test_input_data
        for i in range(len(self.network_weights)):  # whole
            value_layer = np.maximum(np.dot(value_layer, self.network_weights[i]) + self.network_biases[i], 0)
        avg = (np.sum(value_layer, axis=1)).reshape((len(value_layer), 1))
        avg[avg == 0] = 0.001
        value_layer = value_layer / avg
        value = np.sum(np.square(value_layer - self.test_output_data))
        global_test_fitness = value / (self.out_layer * len(self.test_input_data))
        if show:
            value_layer = np.round(value_layer, 2)
            for i in range(len(self.test_input_data)):
                print("{}..".format(str(self.test_input_data[i])[:4]), "::", self.test_output_data[i], "::",
                      value_layer[i])
        return global_test_fitness + self.weight_decay

    def update_weights(self, params, value):
        layer, node, out_node = params
        if out_node == -1:  # if it is a bias
            self.network_biases[layer][0, node] = value
        else:
            self.network_weights[layer][node, out_node] = value
        self.update_weight_decay()

    def test(self, params, value):
        layer, node, out_node = params
        if out_node == -1:
            current_value = self.network_biases[layer][0, node]
            self.network_biases[layer][0, node] = value
            test_value = self.feed_forward()
            self.network_biases[layer][0, node] = current_value
        else:
            current_value = self.network_weights[layer][node, out_node]
            self.network_weights[layer][node, out_node] = value
            test_value = self.feed_forward()
            self.network_weights[layer][node, out_node] = current_value
        return test_value

    def globalTest(self, params, value):
        layer, node, out_node = params
        if out_node == -1:
            current_value = self.network_biases[layer][0, node]
            self.network_biases[layer][0, node] = value
            test_value = self.feed_forward_global()
            self.network_biases[layer][0, node] = current_value
        else:
            current_value = self.network_weights[layer][node, out_node]
            self.network_weights[layer][node, out_node] = value
            test_value = self.feed_forward_global()
            self.network_weights[layer][node, out_node] = current_value
        return test_value
