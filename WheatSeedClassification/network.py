from random import seed
from random import random

import math

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
    #input values
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)

    #hidden layer    
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

seed(1)
#       2 input weights and 1 bias 
#[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
#               weight, bias
#[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}] 2 outputs. number of neurons, each neurons has 1 weight and 1 bias.
network = initialize_network(2, 1, 2) 
for layer in network:
	print(layer)    


#We can break forward propagation down into three parts:
#1. Neuron Activation.
#2. Neuron Transfer.
#3. Forward Propagation.'
#-------------------------------------------------------------------------------
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1] #get the last value
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation    

#--------------------------------------------------------------------------------
# Once a neuron is activated, we need to transfer the activation to see what the neuron output actually is.
# Transfer neuron activation
# sigmoid activation function
def transfer(activation):
	return 1.0 / (1.0 + math.exp(-activation))    

#----------------------------------------------------------------------------------
# Forward propagate input to a network output
# You can see that a neuron’s output value is stored in the neuron with the name ‘output‘. 
# You can also see that we collect the outputs for a layer in an array named new_inputs that becomes the 
# array inputs and is used as inputs for the following layer.
# The function returns the outputs from the last layer also called the output layer.
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs) #return a number
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs   

# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None] #for starter always times 1
output = forward_propagate(network, row)
print(output)     #[0.6629970129852887, 0.7253160725279748]

#-------------------------------------------------------------------------------
# Back propogate error
# Broken down into
# 1. Transfer Derivitive
# 2. Error Backpropagation

# Given an output value from a neuron, we need to calculate it’s slope.
# We are using the sigmoid transfer function, the derivative of which can be calculated as follows:
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))): #start from output layer first
		layer = network[i]
		errors = list()
		if i != len(network)-1: #hidden layer
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else: #output layer
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, 
		{'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)    

#------------------------------------------------------------------------------
# Train Network
# The network is trained using stochastic gradient descent.
# This involves multiple iterations of exposing a training dataset to the network and for each row of data forward propagating the inputs, backpropagating the error and updating the network weights.

# This part is broken down into two sections:

# 1. Update Weights.
# 2. Train Network.

# Once errors are calculated for each neuron in the network via the back propagation method above, they can be used to update weights.

# Learning rate controls how much to change the weight to correct for the error
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1] # get first 2 values
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta'] # add to the last value among the 3, but why?

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row) #sum all weights * input, calculate the activation value for each neuron.
			expected = [0 for i in range(n_outputs)] #
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))				

#Sample dataset to use
# X1			X2			Y
# 2.7810836			2.550537003		0
# 1.465489372		2.362125076		0
# 3.396561688		4.400293529		0
# 1.38807019		1.850220317		0
# 3.06407232		3.005305973		0
# 7.627531214		2.759262235		1
# 5.332441248		2.088626775		1
# 6.922596716		1.77106367		1
# 8.675418651		-0.242068655	1
# 7.673756466		3.508563011		1			

# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1 #x1 and x2 so 2 inputs
n_outputs = len(set([row[-1] for row in dataset])) #get the last column value and put in set (set store only unique values.)
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))	

# Test making predictions with the network
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))