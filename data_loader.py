import math

import numpy


def replace_missing_values(line):
    avg_value = 0
    num_values = 0
    missing_values = []
    for i, val in enumerate(line):
        if val != "?":
            avg_value += float(val)
            num_values += 1
        else:
            missing_values.append(i)
    avg_value = avg_value / num_values
    for index in missing_values:
        line[index] = str(avg_value)
    return line


def get_iris_data():
    file = open("./Data/iris.data", "r")
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


def get_heart_disease():
    file = open("./Data/processed.cleveland.data", "r")
    data = []
    num_input_nodes = 13
    num_hidden_nodes = 6
    num_hidden_layers = 1
    num_output_nodes = 5
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split(",")
        try:
            input_layer = numpy.array([float(x) for x in line[:13]])
        except:
            input_layer = numpy.array([float(x) for x in replace_missing_values(line[:13])])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [0, 0, 0, 0, 0]
        out[int(line[13])] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    for i in range(len(data)):
        for v in range(len(data[i][0])):
            data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("Heart disease data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_wine_data():
    file = open("./Data/wine.data", "r")
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


def get_soybean_large():
    file = open("./Data/soybean-large.data", "r")
    data = []
    num_input_nodes = 35
    num_hidden_nodes = 12
    num_hidden_layers = 1
    num_output_nodes = 19
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split(",")
        try:
            input_layer = numpy.array([float(x) for x in line[1:]])
        except:
            input_layer = numpy.array([float(x) for x in replace_missing_values(line[1:])])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [0 for _ in range(19)]
        out[int(line[0])] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    for i in range(len(data)):
        for v in range(len(data[i][0])):
            data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("Soyabean data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_breast_cancer_data():
    file = open("./Data/wdbc.data", "r")
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
    print("wdbc data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_coil_2000():
    file = open("./Data/ticdata2000.data", "r")
    data = []
    num_input_nodes = 85
    num_hidden_nodes = 85
    num_hidden_layers = 1
    num_output_nodes = 1
    min_value = math.inf
    max_value = -math.inf
    for line in file:
        line = line.strip().split("\t")
        try:
            input_layer = numpy.array([float(x) for x in line[:85]])
        except:
            input_layer = numpy.array([float(x) for x in replace_missing_values(line[:85])])
        min_value = min(numpy.amin(input_layer), min_value)
        max_value = max(numpy.amax(input_layer), max_value)
        out = [float(line[85])]
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    for i in range(len(data)):
        for v in range(len(data[i][0])):
            data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("Coil data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_crime():
    file = open("./Data/communities.data", "r")
    data = []
    num_input_nodes = 122
    num_hidden_nodes = 122
    num_hidden_layers = 1
    num_output_nodes = 1
    # min_value = math.inf
    # max_value = -math.inf
    for line in file:
        line = line.strip().split(",")[5:]
        try:
            input_layer = numpy.array([float(x) for x in line[:122]])
        except:
            input_layer = numpy.array([float(x) for x in replace_missing_values(line[:122])])
        # min_value = min(numpy.amin(input_layer), min_value)
        # max_value = max(numpy.amax(input_layer), max_value)
        out = [float(line[122])]
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    # for i in range(len(data)):
    #     for v in range(len(data[i][0])):
    #         data[i][0][v] = (data[i][0][v] - min_value) / (max_value - min_value)
    print("Crime data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


def get_cnae9_data():
    file = open("./Data/CNAE-9.data")
    data = []
    num_input_nodes = 856
    num_hidden_nodes = 30
    num_hidden_layers = 1
    num_output_nodes = 9
    for line in file:
        line = line.strip().split(",")
        input_layer = numpy.array([float(x) for x in line[1:]])
        out = [0 for _ in range(9)]
        out[int(line[0]) - 1] = 1
        output_layer = numpy.array(out)
        data.append((input_layer, output_layer))
    print("CNAE9 data loaded")
    return (num_input_nodes, num_hidden_nodes, num_hidden_layers, num_output_nodes), data


