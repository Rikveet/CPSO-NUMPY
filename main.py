import copy
import math
from random import Random as Rand

import matplotlib.pyplot as plt
import numpy as np

import data_loader
from Network import Network


class Cpso:

    def __init__(self):
        self.random = Rand()
        self.random.seed(10)
        particles = 20
        network_config, self.data = data_loader.get_breast_cancer_data()
        self.input_nodes = network_config[0]
        self.hidden_nodes = network_config[1]
        self.output_nodes = network_config[3]
        self.contextVector = self.generateOneVec(network_config)
        self.dimension = len(self.contextVector)
        # pprint(self.contextVector)
        self.layers = 2 + network_config[2]  # input layer + hidden layers + output layer
        # decomposition = self.psoDecompose()
        # decomposition = self.layerDecompose()
        # decomposition = self.factorizedLayerDecompose()
        # decomposition = self.nodeDecompose()
        decomposition = self.factorizedNodeDecompose()
        self.swarms = []
        self.swarms = self.createSwarms(decomposition)
        self.net = Network(self.random, network_config, self.data)
        self.iterations = 200
        self.merge_nr = 2
        self.do_merge = False
        self.merge_iter = math.floor(self.iterations / (1 + (math.log(self.dimension) / math.log(self.merge_nr))))

        decomposition_type = "node_factorized"
        self.decom_func = None
        decomDict = {
            "layer": {
                "function": self.layerDecompose,
                "sub_swarm_size": self.layers - 1
            },
            "layer_factorized": {
                "function": self.factorizedLayerDecompose,
                "sub_swarm_size": self.output_nodes + 1
            },
            "node": {
                "function": self.nodeDecompose,
                "sub_swarm_size": self.hidden_nodes + self.output_nodes
            },
            "node_factorized": {
                "function": self.factorizedNodeDecompose,
                "sub_swarm_size": self.hidden_nodes + self.output_nodes
            }
        }
        self.do_decompose = False
        if self.do_decompose and (decomposition_type in decomDict.keys()):
            self.decom_func = decomDict[decomposition_type]["function"]
            self.decompose_iter = math.floor(
                self.iterations /
                (1 + (math.log(self.dimension) / math.log(decomDict[decomposition_type]["sub_swarm_size"]))))
        else:
            self.do_decompose = False
        self.optimize()

    def createWeight(self, index: int, layer: int, node: int, j: int, max_particle_value: float,
                     min_particle_value: float) -> {}:
        """
        This method creates a weight object.
        :param min_particle_value:
        :param max_particle_value:
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
                 "w": self.random.uniform(min_particle_value, max_particle_value),  # current weight
                 "bw": self.random.uniform(min_particle_value, max_particle_value),  # current best weight
                 "f": math.inf,  # personal best fitness.
                 "v": 0
                 } for _ in range(20)]

    def generateOneVec(self, networkConfig: (int, int, int, int)) -> []:
        """
        This method returns a context vector(list), each value in the vector is a weight with layer,node,j, weight value
        and velocity.
        :param networkConfig: a tuple storing network configuration.
        :return: one vector storing all the weights in a network.
        """
        vector = []
        num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes = networkConfig
        # input layer
        min_weight = -1
        max_weight = 1
        min_bias = -0.5
        max_bias = 0.5
        weight_index = 0
        layer = 0
        for i in range(num_input_nodes):
            for j in range(num_hidden_nodes):
                vector.append(
                    self.createWeight(weight_index, layer, i, j, min_weight, max_weight))  # input to hidden weights
                weight_index += 1

        for i in range(num_hidden_nodes):
            vector.append(self.createWeight(weight_index, layer, i, -1, min_bias, max_bias))  # input to hidden bias
            weight_index += 1
        # hidden layers
        layer = 1
        while num_hidden_layers > 1:
            for i in range(num_hidden_nodes):
                vector.append(
                    self.createWeight(weight_index, layer, i, -1, min_bias, max_bias))  # hidden to hidden bias
                weight_index += 1
                for j in range(num_hidden_nodes):
                    vector.append(
                        self.createWeight(weight_index, layer, i, j, min_weight,
                                          max_weight))  # hidden to hidden weights
                    weight_index += 1
            layer += 1
            num_hidden_layers -= 1
        # hidden to output layers
        for i in range(num_hidden_nodes):
            for j in range(num_output_nodes):
                vector.append(
                    self.createWeight(weight_index, layer, i, j, min_weight, max_weight))  # hidden to output weights
                weight_index += 1
        for i in range(num_output_nodes):
            vector.append(self.createWeight(weight_index, layer, i, -1, min_bias, max_bias))  # hidden to output bias
            weight_index += 1
        return vector

    def psoDecompose(self):
        decomposition = []
        cv = self.contextVector
        swarm = []
        for weightParticles in cv:
            swarm.append(copy.deepcopy(weightParticles))
        decomposition.append(swarm)
        return decomposition

    def layerDecompose(self) -> [[]]:
        decomposition = []
        cv = self.contextVector
        # all inputs from hidden to output layer.
        for layer in range(1, self.layers):
            temp = []
            for weightParticles in cv:
                if weightParticles[0]["layer"] == layer - 1:  # input or bias
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)

        return decomposition

    def factorizedLayerDecompose(self):
        decomposition = []
        layers = self.layers
        cv = self.contextVector
        for jNode in range(self.output_nodes):
            temp = []
            # all weights & biases connecting layers in-between input layer and hidden layer.
            for layer in range(1, layers - 1):
                for weightParticles in cv:
                    if weightParticles[0]["layer"] == layer - 1:
                        temp.append(copy.deepcopy(weightParticles))
            # output layer weights input for a node.
            for weightParticles in cv:
                if (weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == jNode) or \
                        (weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["node"] == jNode and
                         weightParticles[0]["j"] == -1):
                    # -2 to get last hidden layer
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        # additional sub swarm for Hidden to output node.
        temp = []
        for weightParticles in cv:
            if weightParticles[0]["layer"] == layers - 2:
                # -2 to get last hidden layer.
                temp.append(copy.deepcopy(weightParticles))
        decomposition.append(temp)
        return decomposition

    def nodeDecompose(self) -> [[]]:
        decomposition = []
        cv = self.contextVector
        layers = self.layers
        # hidden layers
        for layer in range(1, layers - 1):
            for node in range(self.hidden_nodes):
                temp = []
                for weightParticles in cv:
                    if (weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["j"] == node) or (
                            weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["node"] == node and
                            weightParticles[0]["j"] == -1):  # input or bias
                        temp.append(copy.deepcopy(weightParticles))
                decomposition.append(temp)
        # output layer
        for node in range(self.output_nodes):
            temp = []
            for weightParticles in cv:
                if (weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == node) or (
                        weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["node"] == node and
                        weightParticles[0]["j"] == -1):
                    # - 2 to get last hidden layer
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        return decomposition

    def factorizedNodeDecompose(self) -> [[]]:
        decomposition = []
        cv = self.contextVector
        layers = self.layers
        # hidden layers
        for layer in range(1, layers - 1):
            for node in range(self.hidden_nodes):
                temp = []
                for weightParticles in cv:
                    # input weights or biases and output weights
                    if (weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["j"] == node) \
                            or (weightParticles[0]["layer"] == layer - 1 and weightParticles[0]["node"] == node
                                and weightParticles[0]["j"] == -1) \
                            or (weightParticles[0]["layer"] == layer and weightParticles[0]["node"] == node
                                and weightParticles[0]["j"] != -1):
                        temp.append(copy.deepcopy(weightParticles))
                decomposition.append(temp)
        # output layer
        for node in range(self.output_nodes):
            temp = []
            for weightParticles in cv:
                # Input weights or biases
                if (weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["j"] == node) \
                        or (weightParticles[0]["layer"] == layers - 2 and weightParticles[0]["node"] == node
                            and weightParticles[0]["j"] == -1):
                    # -2 to get last hidden layer, -1 for output layer. Weight and bias.
                    temp.append(copy.deepcopy(weightParticles))
            decomposition.append(temp)
        return decomposition

    @staticmethod
    def createSwarms(config: [[]]):
        swarms = []
        for subSwarm in config:
            subSwarmValues = {}
            for weightParticles in subSwarm:
                subSwarmValues[weightParticles[0]["key"]] = {"lbw": 0,  # local best weight
                                                             "f": math.inf}  # local best fitness
            swarms.append({"values": subSwarmValues,
                           "particles": subSwarm})
        return swarms

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
                        swarms[i]["particles"][j][p]["w"] = self.random.uniform(-1, 1)
                        swarms[i]["particles"][j][p]["bw"] = self.random.uniform(-1, 1)
                        swarms[i]["particles"][j][p]["f"] = math.inf
                        swarms[i]["particles"][j][p]["v"] = 0
            self.swarms = swarms

        print("Current swarms :", len(self.swarms))

    def decompose(self):
        decomposition = self.decom_func()
        values = self.swarms[0]["values"]  # global best for a swarm
        swarms = self.createSwarms(decomposition)
        for swarm in swarms:  # keep the global best values.
            for value in swarm["values"]:
                swarm["values"][value] = copy.deepcopy(values[value])
        self.swarms = swarms
        self.do_decompose = False

    def optimize(self):
        max_velocity = 0.14286
        c1 = 1.49618
        c2 = 1.49618
        w = 0.729844
        lda = 0.001  # lambda
        y_values = []
        iterations = self.iterations
        for iteration in range(iterations):
            self.net.create_batch()
            if self.do_merge and iteration % self.merge_iter == 0 and iteration != 0:
                self.merge()
            elif self.do_decompose and iteration % self.decompose_iter == 0 and iteration != 0:
                self.decompose()
            for subSwarm in self.swarms:
                # update best local values and personal values
                for i in range(20):
                    for particle in subSwarm["particles"]:
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
                for i in range(20):
                    for particle in subSwarm["particles"]:
                        w_sum = self.net.get_weight_decay()
                        wd = lda * (w_sum / (w_sum + 1))
                        v = (w * particle[i]["v"]) + (
                                c1 * self.random.uniform(0, 1) * (particle[i]["bw"] - particle[i]["w"])) + \
                            (c2 * self.random.uniform(0, 1) * (
                                    subSwarm["values"][particle[i]["key"]]["lbw"] - particle[i]["w"]))
                        v = min(max(-max_velocity, v), max_velocity)
                        particle[i]["v"] = v
                        particle[i]["w"] += v + wd
                        particle[i]["w"] = min(max(-1, particle[i]["w"]), 1)
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
    np.set_printoptions(suppress=True)
    Cpso()
