import numpy as np
from writeNNet import writeNNet

class Model:
	'''
	Models captured by this class :
		Any number of layers with at least 1 hidden layer
		ReLU after each hidden layer
	'''
	def __init__(self, num_neurons, weights_init=None, biases_init=None):
		'''
		Arguments :
			num_neurons : list of neuron count of each layer
				(First value in list should be dim of input)
		Functionality :
			Initializes weights and biases of the network to
			random gaussian initialization.
			TODO : If weights_init and/or biases_init given, then
			initialize using the given values.
			TODO(if needed) : Different kind of random init
		'''
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)
		'''
		if weights_init is not None:
			self.weights = ...
			if biases_init is None:
				self.biases = zeros
			else:
				self.biases = ...
		else:
		'''
		self.weights = {}
		self.biases = {}

		for lyr in range(1, self.num_layers):
			W_ = np.random.randn(num_neurons[lyr], num_neurons[lyr-1])
			b_ = np.random.randn(num_neurons[lyr])
			self.weights[lyr] = W_
			self.biases[lyr] = b_

		self.max_num_hidden = -1
		for lyr in range(1, self.num_layers-1):
			if self.max_num_hidden < num_neurons[lyr]:
				self.max_num_hidden = num_neurons[lyr]

	def forward(self, input_val):
		'''
		Arguments :
			input_val : 1d np array of shape (self.num_neurons[0],)
		Returns :
			Logits as output of model
		'''
		assert input_val.shape == (self.num_neurons[0],), "Shape of input wrong"
		curr = input_val
		for lyr in range(1, self.num_layers):
			curr = np.matmul(self.weights[lyr], curr) + self.biases[lyr]
			if lyr < self.num_layers-1: # since no ReLU on last layer
				curr = np.maximum(curr, 0)
		return curr

	def activation_pattern_from_input(self, input_val):
		'''
		Arguments :
			input_val : 1d np array of shape (self.num_neurons[0],)
		Returns :
			Full activation pattern as a 2d np array
			of shape (self.num_layers-2, self.max_num_hidden)
			Convention is 0 : off, 1 : on, 2 : nonexistent neuron
		'''
		assert input_val.shape == (self.num_neurons[0],), "Shape of input wrong"
		act_pattern = np.zeros((self.num_layers-2, self.max_num_hidden))
		curr = input_val
		for lyr in range(1, self.num_layers-1):
			curr = np.matmul(self.weights[lyr], curr) + self.biases[lyr]
			curr = np.maximum(curr, 0)
			act_pattern_ = np.where(curr > 0, 1.0, 0.0)
			append_ = 2.0*np.ones(self.max_num_hidden - self.num_neurons[lyr])
			act_pattern[lyr-1] = np.concatenate((act_pattern_, append_))
		return act_pattern

	def write_NNET(self, fname):
		'''
		Arguments :
			fname : file to write the nnet
		'''
		weights_list = []
		biases_list = []
		for lyr in range(1, self.num_layers):
			weights_list.append(self.weights[lyr])
			biases_list.append(self.biases[lyr])
		num_inp = self.num_neurons[0]
		writeNNet(weights_list, biases_list, [-3.0]*num_inp,
			[3.0]*num_inp, [0.0]*(num_inp+1),[1.0]*(num_inp+1),fname)

	def affine_params(self, input_val):
		'''
		Arguments :
			input_val : 1d np array of shape (self.num_neurons[0],)
		Returns :
			weights and biases for final logits w.r.to generic input x,
			for the activation pattern described by input_val
		'''
		assert input_val.shape == (self.num_neurons[0],), "Shape of input wrong"
		weight = self.weights[1]
		bias = self.biases[1]
		for lyr in range(2, self.num_layers):
			curr = np.matmul(weight, input_val) + bias
			curr = np.maximum(curr, 0)
			nz_idx = np.argwhere(curr).squeeze(axis=1)
			nz_count = nz_idx.shape[0]

			weight_mask = np.zeros_like(weight)
			weight_mask[nz_idx] = np.ones((nz_count, weight.shape[1]))
			bias_mask = np.zeros_like(bias)
			bias_mask[nz_idx] = np.ones((nz_count,))
			
			weight_ = weight*weight_mask
			bias_ = bias*bias_mask
			weight = np.matmul(self.weights[lyr], weight_)
			bias = np.matmul(self.weights[lyr], bias_) + self.biases[lyr]
		return weight, bias

# Basic testing code
# Below code runs without errors
'''
num_neurons = [2,3,3,2]
model = Model(num_neurons)
model.write_NNET('dummy.nnet')
for r in range(10):
	print("On random input " + str(r))
	inp = np.random.randn(2)
	model.activation_pattern_from_input(inp)
	model.affine_params(inp)
'''