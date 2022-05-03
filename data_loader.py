import math

import numpy


def get_iris_data():
    file = open("Data/iris.data", "r")
    data = []
    num_input_nodes = 4
    num_hidden_nodes = 4
    num_hidden_layers = 1
    num_output_nodes = 3
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split(",")
        input_layer = numpy.array([float(x) for x in line[0:4]])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [0, 0, 0]
        out[int(line[4])] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    for i in range(len(data)):
        for v in range(len(data[i][0])):
            data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("iris data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_mist100_data():
    file = open("Data/mist100.data")


def get_cnae9_data():
    file = open("Data/CNAE-9.data")
    data = []
    num_input_nodes = 856
    num_hidden_nodes = 30
    num_hidden_layers = 1
    num_output_nodes = 9
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split(",")
        input_layer = numpy.array([float(x) for x in line[1:]])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [0 for _ in range(9)]
        out[int(line[0]) - 1] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    print("CNAE9 data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_wine_data():
    file = open("Data/wine.data", "r")
    data = []
    num_input_nodes = 13
    num_hidden_nodes = 10
    num_hidden_layers = 1
    num_output_nodes = 3
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split(",")
        input_layer = numpy.array([float(x) for x in line[1:]])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [0, 0, 0]
        out[int(line[0]) - 1] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    for i in range(len(data)):
        for v in range(len(data[i][0])):
            data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("wine data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_breast_cancer_data():
    file = open("Data/wdbc.data", "r")
    data = []
    num_input_nodes = 30
    num_hidden_nodes = 25
    num_hidden_layers = 1
    num_output_nodes = 2
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split(",")
        input_layer = numpy.array([float(x) for x in line[2:]])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [0, 0]
        if line[1] == "M":
            out[0] = 1
        else:
            out[1] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    for i in range(len(data)):
        for v in range(len(data[i][0])):
            data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("Breast cancer data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data
