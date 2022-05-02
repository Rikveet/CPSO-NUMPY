# # import copy
# # import math
# # import random
# #
# # import data_loader
# # from Network import Network
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# #
# # # from pprint import pprint
# #
# #
# # def createWeight(index: int, layer: int, node: int, j: int) -> {}:
# #     """
# #     This method creates a weight object.
# #     :param index: key for accessing global values such as weight and fitness.
# #     :param layer: layer in which node is present
# #     :param node: the node for which weight is being generated
# #     :param j: the node to which the weight is connected in the next layer
# #     :return: A dictionary of layer, node, j and initialized weight with range -1,1 and 0 velocity.
# #     """
# #     return [{"key": index,
# #              "layer": layer,
# #              "node": node,
# #              "j": j,
# #              "w": random.uniform(-1, 1),  # current weight
# #              "bw": random.uniform(-1, 1),  # current best weight
# #              "f": math.inf,  # personal best fitness.
# #              "v": 0
# #              } for _ in range(20)]
# #
# #
# # def generateOneVec(networkConfig: (int, int, int, int)) -> []:
# #     """
# #     This method returns a context vector(list), each value in the vector is a weight with layer,node,j, weight value
# #     and velocity.
# #     :param networkConfig: a tuple storing network configuration.
# #     :return: one vector storing all the weights in a network.
# #     """
# #     vector = []
# #     num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes = networkConfig
# #     # input layer
# #     weight_index = 0
# #     layer = 0
# #     for i in range(num_input_nodes):
# #         for j in range(num_hidden_nodes):
# #             vector.append(createWeight(weight_index, layer, i, j))
# #             weight_index += 1
# #     layer = 1
# #     while num_hidden_layers > 1:
# #         for i in range(num_hidden_nodes):
# #             for j in range(num_hidden_nodes):
# #                 vector.append(createWeight(weight_index, layer, i, j))
# #                 weight_index += 1
# #         layer += 1
# #     for i in range(num_hidden_nodes):
# #         for j in range(num_output_nodes):
# #             vector.append(createWeight(weight_index, layer, i, j))
# #             weight_index += 1
# #     return vector
# #
# #
# # class Cpso:
# #
# #     def __init__(self):
# #         random.seed(10)
# #         particles = 20
# #         network_config, self.data = data_loader.get_iris_data()
# #         self.contextVector = generateOneVec(network_config)
# #         # pprint(self.contextVector)
# #         layers = 2 + network_config[2]  # input layer + hidden layers + output layer
# #         # decomposition = self.layerDecompose(1, layers)
# #         decomposition = self.factorizedLayerDecompose(1, network_config[3], layers)
# #         # decomposition = self.nodeDecompose(network_config[1], network_config[3], layers)
# #         # decomposition = self.factorizedNodeDecompose(network_config[1], network_config[3], layers)
# #         self.swarms = []
# #         self.createSwarms(decomposition)
# #         self.net = Network(network_config, self.data)
# #         self.optimize()
# #
# #     def layerDecompose(self, startLayer: int, layers: int) -> [[]]:
# #         decomposition = []
# #         cv = self.contextVector
# #         # all inputs from hidden to output layer.
# #         for layer in range(startLayer, layers):
# #             temp = []
# #             for weightParticles in cv:
# #                 if weightParticles[0]["layer"] == layer - 1:
# #                     temp.append(copy.deepcopy(weightParticles))
# #             decomposition.append(temp)
# #         return decomposition
# #
# #     def factorizedLayerDecompose(self, startLayer: int, num_output_nodes: int, layers: int):
# #         decomposition = []
# #         cv = self.contextVector
# #         for jNode in range(num_output_nodes):
# #             temp = []
# #             # all weights connecting layers in-between input layer and hidden layer.
# #             for layer in range(startLayer, layers - 1):
# #                 for weightParticles in cv:
# #                     if weightParticles[0]["layer"] == layer - 1:
# #                         temp.append(copy.deepcopy(weightParticles))
# #             # output layer weights input for a node.
# #             for weightParticles in cv:
# #                 if weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == jNode:
# #                     # - 2 to get last hidden layer
# #                     temp.append(copy.deepcopy(weightParticles))
# #             decomposition.append(temp)
# #         return decomposition
# #
# #     def nodeDecompose(self, num_hidden_nodes: int, num_output_nodes: int, layers: int) -> [[]]:
# #         decomposition = []
# #         cv = self.contextVector
# #         # hidden layers
# #         for layer in range(1, layers - 1):
# #             for node in range(num_hidden_nodes):
# #                 temp = []
# #                 for weightParticles in cv:
# #                     if weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["j"] == node:
# #                         temp.append(copy.deepcopy(weightParticles))
# #                 decomposition.append(temp)
# #         # output layer
# #         for node in range(num_output_nodes):
# #             temp = []
# #             for weightParticles in cv:
# #                 if weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == node:
# #                     # - 2 to get last hidden layer
# #                     temp.append(copy.deepcopy(weightParticles))
# #             decomposition.append(temp)
# #         return decomposition
# #
# #     def factorizedNodeDecompose(self, num_hidden_nodes: int, num_output_nodes: int, layers: int) -> [[]]:
# #         decomposition = []
# #         cv = self.contextVector
# #         # hidden layers
# #         for layer in range(1, layers - 1):
# #             for node in range(num_hidden_nodes):
# #                 temp = []
# #                 for weightParticles in cv:
# #                     if (weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["j"] == node) or \
# #                             (weightParticles[0]["layer"] == layer and weightParticles[0]["node"] == node):
# #                         temp.append(copy.deepcopy(weightParticles))
# #                 decomposition.append(temp)
# #         # output layer
# #         for node in range(num_output_nodes):
# #             temp = []
# #             for weightParticles in cv:
# #                 if weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == node:
# #                     # - 2 to get last hidden layer
# #                     temp.append(copy.deepcopy(weightParticles))
# #             decomposition.append(temp)
# #         return decomposition
# #
# #     def createSwarms(self, config: [[]]) -> [[]]:
# #         swarms = []
# #         for subSwarm in config:
# #             subSwarmValues = {}
# #             for weightParticles in subSwarm:
# #                 subSwarmValues[weightParticles[0]["key"]] = {"lbw": 0,  # local best weight
# #                                                              "f": math.inf}  # local best fitness
# #             swarms.append({"values": subSwarmValues,
# #                            "particles": subSwarm})
# #         self.swarms = swarms
# #
# #     def merge(self):
# #         # pass
# #         if len(self.swarms) > 1:
# #             swarms = []
# #             for i in range(0, len(self.swarms), 2):
# #                 if i + 1 < len(self.swarms):
# #                     subSwarms = {"values": self.swarms[i]["values"] | self.swarms[i + 1]["values"],
# #                                  "particles": self.swarms[i]["particles"] + self.swarms[i + 1]["particles"]}
# #                     swarms.append(subSwarms)
# #             self.swarms = swarms
# #         print("Current swarms :", len(self.swarms))
# #
# #     def optimize(self):
# #         iterations = 100
# #         max_velocity = 0.14286
# #         c1 = 1.49618
# #         c2 = 1.49618
# #         w = 0.729844
# #         lda = 0.001  # lambda
# #         y_values = []
# #         merge_iter = 20
# #         for iteration in range(iterations):
# #             self.net.create_batch()
# #             if iteration % 34 == 0 and iteration != 0:
# #                 self.merge()
# #             for subSwarm in self.swarms:
# #                 # update best local values and personal values
# #                 for particle in subSwarm["particles"]:
# #                     for i in range(20):
# #                     for weight in particle:
# #                         node_params = (weight["layer"], weight["node"], weight["j"])
# #                         fitness = self.net.test(node_params, weight["w"])
# #                         if fitness <= weight["f"]:
# #                             weight["bw"] = weight["w"]
# #                             weight["f"] = fitness
# #                         if fitness <= subSwarm["values"][weight["key"]]["f"]:
# #                             subSwarm["values"][weight["key"]]["lbw"] = weight["w"]
# #                             subSwarm["values"][weight["key"]]["f"] = fitness
# #                             self.net.update_weights(node_params, weight["w"])
# #                 # update velocity and position
# #                 for particle in subSwarm["particles"]:
# #                     w_sum = self.net.get_weight_decay()
# #                     wd = lda * (w_sum / (w_sum + 1))
# #                     for weight in particle:
# #                         v = (w * weight["v"]) + (c1 * random.uniform(0, 1) * (weight["bw"] - weight["w"])) + \
# #                             (c2 * random.uniform(0, 1) * (subSwarm["values"][weight["key"]]["lbw"] - weight["w"]))
# #                         v = min(max(-max_velocity, v), max_velocity)
# #                         weight["v"] = v
# #                         weight["w"] += v + wd
# #             fitness = self.net.feed_forward()
# #             print("Iteration", iteration, " Fitness: ", fitness)
# #             y_values.append(fitness)
# #         print("Training Complete, output:->")
# #         plt.axis([0, iterations, 0, max(y_values)])
# #         plt.plot(np.array(y_values))
# #         plt.show()
# #         self.net.feed_forward(True)
# #
# #
# # if __name__ == "__main__":
# #     Cpso()
#
#
# from random import uniform, shuffle
#
# import numpy
# import numpy as np
#
#
# class Network:
#     def __init__(self, nodes_config, data):
#         self.train_output_data = None
#         self.train_input_data = None
#         input_layer_nodes, hidden_layer_nodes, num_hidden_layers, output_layer_nodes = nodes_config
#         self.out_layer = output_layer_nodes
#         max_weight, min_weight = (-1,1)
#         self.data = data
#         self.layers = []
#
#         self.biasLayers = [numpy.array([0 for _ in range(input_layer_nodes)])]
#         for _ in range(num_hidden_layers):
#             val = [0 for _ in range(hidden_layer_nodes)]
#             self.biasLayers.append(numpy.array(val))
#
#         self.layers.append(numpy.array([[uniform(min_weight, max_weight) for _ in range(hidden_layer_nodes)] for _ in
#                                         range(input_layer_nodes)]))
#         for _ in range(num_hidden_layers - 1):
#             self.layers.append(numpy.array([[uniform(min_weight, max_weight) for _ in range(hidden_layer_nodes)] for _
#                                             in range(hidden_layer_nodes)]))
#         self.layers.append(numpy.array([[uniform(min_weight, max_weight) for _ in range(output_layer_nodes)]
#                                         for _ in range(hidden_layer_nodes)]))
#         np.set_printoptions(precision=2)
#         print("Network initialized")
#
#     def shuffle_data(self):
#         shuffle(self.data)
#
#     def create_batch(self):
#         self.shuffle_data()
#         input_data = []
#         output_data = []
#         for i in range(100):
#             inpt, outpt = self.data[i]
#             input_data.append(inpt)
#             output_data.append(outpt)
#         self.train_input_data = numpy.array(input_data)
#         self.train_output_data = numpy.array(output_data)
#
#     def get_weight_decay(self):
#         wd = 0
#         for layer in self.layers:
#             wd += np.sum(np.square(layer))
#         return wd
#
#     def feed_forward(self, show=False):
#         value = 0
#         value_layer = self.train_input_data
#         for i in range(len(self.layers) - 1):
#             value_layer = numpy.maximum(numpy.matmul(value_layer, self.layers[i]) + self.biasLayers[i], 0)  #
#         value_layer = numpy.maximum(numpy.matmul(value_layer, self.layers[-1]), 0)
#         value += np.sum(np.square(value_layer - self.train_output_data))
#         if show:
#             for i in range(100):
#                 print(self.train_input_data[i], "::", self.train_output_data[i], "::", value_layer[i])  # )
#         return value / (self.out_layer * 100)
#
#     def update_weights(self, params, value):
#         layer, node, out_node = params
#         self.layers[layer][node, out_node] = value
#
#     def update_bias(self, params, value):
#         layer, node = params
#         self.biasLayers[layer][node] = value
#
#     def test(self, params, value):
#         layer, node, out_node = params
#         current_value = self.layers[layer][node, out_node]
#         self.layers[layer][node, out_node] = value
#         test_value = self.feed_forward()
#         self.layers[layer][node, out_node] = current_value
#         return test_value
#
#     def test_bias(self, params, value):
#         layer, node = params
#         current_value = self.biasLayers[layer][node]
#         self.biasLayers[layer][node] = value
#         test_value = self.feed_forward()
#         self.biasLayers[layer][node] = current_value
#         return test_value
#
#
#
#
#
#
