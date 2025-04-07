import numpy as np
import nnfs # neural network from scratch og library
from nnfs.datasets import spiral_data # need to specify import from library else overrides everything (if wildcard is used)
import matplotlib.pyplot as plt

# nnfs.init does 3 things: sets random seed to 0, 
# creates a float32 dtype default,
# overrides the original dot product from NumPy
# using nnfs will keep the random to be the same for learning
nnfs.init()

# X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

#print(np.random.rand(2,5))

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # takes array shape and return with 0's

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

# create datasets
X, y = spiral_data(samples=100, classes=3)

# create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# create second Dense layer with 3 input features (as we take output of previous layer here)
# and 3 output values
dense2 = Layer_Dense(3, 3)

# create Softmax Activation (to be used with Dense Layer)
activation2 = Activation_Softmax()

# perform a forward pass of our training data through this layer
dense1.forward(X)

# forward pass through activation fucn.
# takes in ioutput from previous layer
activation1.forward(dense1.output)

# forward pass through 2nd Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# forward pass through activation func.
# takes the output of 2nd dense layer here
activation2.forward(dense2.output)

# print(dense1.output[:5])
print(activation2.output[:5])

