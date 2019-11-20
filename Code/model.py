import numpy as np

class Model:
	'''
	Model can be :
		Any number of layers
		ReLU after each hidden layer
		Output is the logits
	'''
	def __init__(self, num_neurons):
		'''
		Arguments :
			num_neurons : list of neuron count of each layer
				(First value in list should be dim of input)
		Returns :
			<Nothing>
			Initializes weights and biases of the network
			TODO(if needed) : Different kinds of initialization
		'''
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)

		self.weights = {}
		self.biases = {}

		for lyr in range(1, self.num_layers):
			# Random Gaussian initialization for now
			W_ = np.random.randn(num_neurons[lyr-1], num_neurons[lyr])
			b_ = np.random.randn(num_neurons[lyr])
			self.weights[lyr] = W_
			self.biases[lyr] = b_

	def activation_pattern_from_input(self, input_val):
		'''
		Arguments :
			input_val : 1d np.array of shape num_neurons[0]
		Returns :
			Full activation pattern as a np array
		'''
		assert input_val.shape == (self.num_neurons[0],), "Shape of input wrong"
		curr = input_val
		for lyr in range(1, self.num_layers):
			self.weights[lyr]


	def write_NNET(self):

	def weights_final(self, input_val):

num_neurons = [2,3,3,2]
model = Model(num_neurons)