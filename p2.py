import numpy as np

# ------ Single Neuron with Numpy ------ #
# inputs = [1.0, 2.0, 3.0, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2.0

# # dot product uses np library, and access with .dot
# outputs = np.dot(weights, inputs) + bias

# print(outputs)
# -------------------------------------- #

# -------- A layer of Neurons with Numpy ------#
# inputs = [1.0, 2.0, 3.0, 2.5]
# weights = [[0.2, 0.8, -0.5, 1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# biases = [2.0, 3.0, 0.5]

# layer_outputs = np.dot(weights, inputs) + biases

# print(layer_outputs)
# -------------------------------------- #

# ------- basic array/matrice with NumPy ---- #
 
# np.array([1, 2, 3]) 
# or
# a = [1, 2, 3]
# np.array([a])

# a = [1, 2, 3]
# b = [2, 3, 4]

# a = np.array([a])
# b = np.array([b]).T
# print(np.dot(a, b))
# ------------------------------------------- #

# Implementation of the above --------------- #
# with 1 hidden column of hidden layer 
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 2.5, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases = [2.0, 3.0, 0.5]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

print(np.exp(-np.inf), np.exp(0))