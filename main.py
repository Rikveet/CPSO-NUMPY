import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

import data_loader
from Network import Network


# from pprint import pprint


def createWeight(index: int, layer: int, node: int, j: int) -> {}:
    """
    This method creates a weight object.
    :param index: key for accessing global values such as weight and fitness.
    :param layer: layer in which node is present
    :param node: the node for which weight is being generated
    :param j: the node to which the weight is connected in the next layer
    :return: A dictionary of layer, node, j and initialized weight with range -1,1 and 0 velocity.
    """
    return [{"key": index,
             "layer": layer,
             "node": node,
             "j": j,
             "w": random.uniform(-1, 1),  # current weight
             "bw": random.uniform(-1, 1),  # current best weight
             "f": math.inf,  # personal best fitness.
             "v": 0
             } for _ in range(20)]


def generateOneVec(networkConfig: (int, int, int, int)) -> []:
    """
    This method returns a context vector(list), each value in the vector is a weight with layer,node,j, weight value
    and velocity.
    :param networkConfig: a tuple storing network configuration.
    :return: one vector storing all the weights in a network.
    """
    vector = []
    num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes = networkConfig
    # input layer
    weight_index = 0
    layer = 0
    for i in range(num_input_nodes):
        for j in range(num_hidden_nodes):
            vector.append(createWeight(weight_index, layer, i, j))
            weight_index += 1
    layer = 1
    while num_hidden_layers > 1:
        for i in range(num_hidden_nodes):
            vector.append(createWeight(weight_index, layer, i, -1))  # bias
            weight_index += 1
            for j in range(num_hidden_nodes):
                vector.append(createWeight(weight_index, layer, i, j))  # weights
                weight_index += 1
        layer += 1
        num_hidden_layers -= 1
    for i in range(num_hidden_nodes):
        vector.append(createWeight(weight_index, layer, i, -1))
        weight_index += 1
        for j in range(num_output_nodes):
            vector.append(createWeight(weight_index, layer, i, j))
            weight_index += 1
    layer += 1
    # output layer biases
    for i in range(num_output_nodes):
        vector.append(createWeight(weight_index, layer, i, -1))
        weight_index += 1
    return vector


class Cpso:

    def __init__(self):
        random.seed(10)
        particles = 20
        network_config, self.data = data_loader.get_iris_data()
        self.contextVector = generateOneVec(network_config)
        self.dimension = len(self.contextVector)
        # pprint(self.contextVector)
        layers = 2 + network_config[2]  # input layer + hidden layers + output layer
        # decomposition = self.layerDecompose(1, layers)
        # decomposition = self.factorizedLayerDecompose(1, network_config[3], layers)
        decomposition = self.nodeDecompose(network_config[1], network_config[3], layers)
        # decomposition = self.factorizedNodeDecompose(network_config[1], network_config[3], layers)
        self.swarms = []
        self.createSwarms(decomposition)
        self.net = Network(network_config, self.data)
        self.optimize()

    def psoDecompose(self):
        pass

    def layerDecompose(self, startLayer: int, layers: int) -> [[]]:
        decomposition = []
        cv = self.contextVector
        # all inputs from hidden to output layer.
        for layer in range(startLayer, layers):
            temp = []
            for weightParticles in cv:
                if weightParticles[0]["layer"] == layer - 1:
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        return decomposition

    def factorizedLayerDecompose(self, startLayer: int, num_output_nodes: int, layers: int):
        decomposition = []
        cv = self.contextVector
        for jNode in range(num_output_nodes):
            temp = []
            # all weights connecting layers in-between input layer and hidden layer.
            for layer in range(startLayer, layers - 1):
                for weightParticles in cv:
                    if weightParticles[0]["layer"] == layer - 1:
                        temp.append(copy.deepcopy(weightParticles))
            # output layer weights input for a node.
            for weightParticles in cv:
                if weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == jNode:
                    # - 2 to get last hidden layer
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        return decomposition

    def nodeDecompose(self, num_hidden_nodes: int, num_output_nodes: int, layers: int) -> [[]]:
        decomposition = []
        cv = self.contextVector
        # hidden layers
        for layer in range(1, layers - 1):
            for node in range(num_hidden_nodes):
                temp = []
                for weightParticles in cv:
                    if weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["j"] == node:
                        temp.append(copy.deepcopy(weightParticles))
                decomposition.append(temp)
        # output layer
        for node in range(num_output_nodes):
            temp = []
            for weightParticles in cv:
                if weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == node:
                    # - 2 to get last hidden layer
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        return decomposition

    def factorizedNodeDecompose(self, num_hidden_nodes: int, num_output_nodes: int, layers: int) -> [[]]:
        decomposition = []
        cv = self.contextVector
        # hidden layers
        for layer in range(1, layers - 1):
            for node in range(num_hidden_nodes):
                temp = []
                for weightParticles in cv:
                    if (weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["j"] == node) or \
                            (weightParticles[0]["layer"] == layer and weightParticles[0]["node"] == node):
                        temp.append(copy.deepcopy(weightParticles))
                decomposition.append(temp)
        # output layer
        for node in range(num_output_nodes):
            temp = []
            for weightParticles in cv:
                if weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == node:
                    # - 2 to get last hidden layer
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        return decomposition

    def createSwarms(self, config: [[]]) -> [[]]:
        swarms = []
        for subSwarm in config:
            subSwarmValues = {}
            for weightParticles in subSwarm:
                subSwarmValues[weightParticles[0]["key"]] = {"lbw": 0,  # local best weight
                                                             "f": math.inf}  # local best fitness
            swarms.append({"values": subSwarmValues,
                           "particles": subSwarm})
        self.swarms = swarms

    def merge(self):
        if len(self.swarms) > 1:
            swarms = []
            for i in range(0, len(self.swarms), 2):
                if i + 1 < len(self.swarms):
                    common_keys = self.swarms[i]["values"].keys() & self.swarms[i + 1]["values"].keys()
                    for key in common_keys:
                        def del_key(index: int, k):
                            del self.swarms[index]["values"][k]
                            self.swarms[index]["particles"] = [self.swarms[index]["particles"][particle_index]
                                                               for particle_index in range(len(self.swarms[index]
                                                                                               ["particles"]))
                                                               if self.swarms[index]["particles"][particle_index][0]
                                                               ["key"] != k]

                        if self.swarms[i]["values"][key]["f"] < self.swarms[i + 1]["values"][key]["f"]:
                            del_key(i + 1, key)
                        else:
                            del_key(i, key)
                    subSwarms = {"values": self.swarms[i]["values"] | self.swarms[i + 1]["values"],
                                 "particles": self.swarms[i]["particles"] + self.swarms[i + 1]["particles"]}
                    swarms.append(subSwarms)
                else:
                    swarms.append(self.swarms[i])
            for i in range(len(swarms)):
                for j in range(len(swarms[i]["particles"])):
                    for p in range(len(swarms[i]["particles"][j])):
                        # reset        "w"  # current weight
                        #              "bw" # current best weight
                        #              "f"  # personal best fitness.
                        #              "v"  # velocity.
                        swarms[i]["particles"][j][p]["w"] = random.uniform(-1, 1)
                        swarms[i]["particles"][j][p]["bw"] = random.uniform(-1, 1)
                        swarms[i]["particles"][j][p]["f"] = math.inf
                        swarms[i]["particles"][j][p]["v"] = 0
            self.swarms = swarms

        print("Current swarms :", len(self.swarms))

    def decompose(self):

        pass

    def optimize(self):
        iterations = 100
        max_velocity = 0.14286
        c1 = 1.49618
        c2 = 1.49618
        w = 0.729844
        lda = 0.001  # lambda
        y_values = []
        n_r = 2
        do_merge = True
        merge_iter = math.floor(iterations / (1 + (math.log(self.dimension) / math.log(n_r))))
        decompose_iter = 0
        for iteration in range(iterations):
            self.net.create_batch()
            if do_merge and iteration % merge_iter == 0 and iteration != 0:
                self.merge()
            for subSwarm in self.swarms:
                # update best local values and personal values
                for particle in subSwarm["particles"]:
                    for i in range(20):
                        node_params = (particle[i]["layer"], particle[i]["node"], particle[i]["j"])
                        fitness = self.net.test(node_params, particle[i]["w"])
                        if fitness <= particle[i]["f"]:
                            particle[i]["bw"] = particle[i]["w"]
                            particle[i]["f"] = fitness
                        if fitness <= subSwarm["values"][particle[i]["key"]]["f"]:
                            subSwarm["values"][particle[i]["key"]]["lbw"] = particle[i]["w"]
                            subSwarm["values"][particle[i]["key"]]["f"] = fitness
                            self.net.update_weights(node_params, particle[i]["w"])
                # update velocity and position
                for particle in subSwarm["particles"]:
                    w_sum = self.net.get_weight_decay()
                    wd = lda * (w_sum / (w_sum + 1))
                    for i in range(20):
                        v = (w * particle[i]["v"]) + (
                                c1 * random.uniform(0, 1) * (particle[i]["bw"] - particle[i]["w"])) + \
                            (c2 * random.uniform(0, 1) * (
                                    subSwarm["values"][particle[i]["key"]]["lbw"] - particle[i]["w"]))
                        v = min(max(-max_velocity, v), max_velocity)
                        particle[i]["v"] = v
                        particle[i]["w"] += v + wd
            fitness = self.net.feed_forward()
            print("Iteration", iteration, " Fitness: ", fitness)
            y_values.append(fitness)
        print("Training Complete, output:->")
        plt.axis([0, iterations, 0, max(y_values)])
        plt.plot(np.array(y_values))
        plt.show()
        final_fitness = self.net.feed_forward(True)
        print("Final fitness:", final_fitness)


if __name__ == "__main__":
    Cpso()
