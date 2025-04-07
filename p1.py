inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0]
            [0.5, -0.91, 0.26, -0.5]
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# ## original, complex and hard to scale ##
# outputs = [
#         # Neuron 1
#         inputs[0]*weights1[0] +
#         inputs[1]*weights1[1] +
#         inputs[2]*weights1[2] +
#         inputs[3]*weights1[3] + bias1,

#         # Neuron 2  
#         inputs[0]*weights2[0] +
#         inputs[1]*weights2[1] +
#         inputs[2]*weights2[2] +
#         inputs[3]*weights2[3] + bias2,

#         # Neuron 3  
#         inputs[0]*weights3[0] +
#         inputs[1]*weights3[1] +
#         inputs[2]*weights3[2] +
#         inputs[3]*weights3[3] + bias3,
#         ]

# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input*weight
    # Add bias
    neuron_output += neuron_bias 
    # Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

print(layer_outputs)