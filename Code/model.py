import numpy as np
from writeNNet import writeNNet

# Two useful functions to check correctness of initializations passes to Model
def weights_correctness(weights_init, num_neurons):
	err_msg = "weights_init is inconsistent with num_neurons"
	num_layers = len(num_neurons)
	for lyr in range(1, num_layers):
		assert weights_init[lyr].shape == (num_neurons[lyr], num_neurons[lyr-1]), err_msg

def biases_correctness(biases_init, num_neurons):
	err_msg = "biases_init is inconsistent with num_neurons"
	num_layers = len(num_neurons)
	for lyr in range(1, num_layers):
		assert biases_init[lyr].shape == (num_neurons[lyr],), err_msg

class Model:
	'''
	Models captured by this class :
		Any number of layers with at least 1 hidden layer
		ReLU after each hidden layer
	'''
	def __init__(self, num_neurons, weights_init=None, biases_init=None,
				min_input_val=-3.0, max_input_val=3.0):
		'''
		Arguments :
			num_neurons : list of neuron count of each layer
				(First value in list should be dim of input)
		Functionality :
			Initializes weights and biases of the network to
			random gaussian initialization.
			TODO(if needed) : Different kind of random init
		'''
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)
		self.min_input_val = min_input_val
		self.max_input_val = max_input_val

		self.max_num_hidden = -1
		self.hidden_neurons = 0
		for lyr in range(1, self.num_layers-1):
			if self.max_num_hidden < num_neurons[lyr]:
				self.max_num_hidden = num_neurons[lyr]
			self.hidden_neurons += num_neurons[lyr]

		self.weights = {}
		self.biases = {}
		if weights_init is not None:
			weights_correctness(weights_init, num_neurons)
			self.weights = weights_init
			if biases_init is None:
				self.biases = {}
				for lyr in range(1, self.num_layers):
					self.biases[lyr] = np.zeros(num_neurons[lyr]) 
			else:
				biases_correctness(biases_init, num_neurons)
				self.biases = biases_init
		else:
			for lyr in range(1, self.num_layers):
				W_ = np.random.randn(num_neurons[lyr], num_neurons[lyr-1])
				b_ = np.random.randn(num_neurons[lyr])
				self.weights[lyr] = W_
				self.biases[lyr] = b_

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
		mean_input_val = 0
		range_input_val = 1
		writeNNet(weights_list, biases_list, [self.min_input_val]*num_inp,
			[self.max_input_val]*num_inp, [mean_input_val]*(num_inp+1),
			[range_input_val]*(num_inp+1), fname)

	def affine_params(self, input_val, flag='output'):
		'''
		Arguments :
			input_val : 1d np array of shape (self.num_neurons[0],)
		Returns :
			If flag is 'output' (or not given) then returns the
			weights and biases for final logits w.r.to generic input x,
			for the activation pattern described by input_val.
			Else, returns the same thing for all layers(hidden + output) in a list.
		'''
		assert input_val.shape == (self.num_neurons[0],), "Shape of input wrong"
		weight = self.weights[1]
		bias = self.biases[1]
		if flag != 'output':
			weights_list = [np.copy(weight)]
			biases_list = [np.copy(bias)]
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
			if flag != 'output':
				weights_list.append(np.copy(weight))
				biases_list.append(np.copy(bias))
		if flag != 'output':
			return weights_list, biases_list
		else:
			return weight, bias

	def affine_params_from_act(self, act_pattern, flag='output'):
		'''
		Arguments :
			act_pattern : 2d np array of shape (self.num_layers-2, self.max_num_hidden)
			Convention is 0 : off, 1 : on, 2 : nonexistent neuron
		Returns :
			If flag is 'output' (or not given) then returns the
			weights and biases for final logits w.r.to generic input x,
			for the activation pattern described by input_val.
			Else, returns the same thing for all layers(hidden + output) in a list.
		'''
		# TODO(if needed)
		pass

# Basic testing code
# Below code runs without errors
'''
num_neurons = [2,2,2]
weights_init = {}
weights_init[1] = np.array([[1.0, -1.0], [-1.0, 1.0]])
weights_init[2] = np.array([[1.0, 0.0], [0.0, 1.0]])
model = Model(num_neurons, weights_init)
model.write_NNET('dummy_init.nnet')
for r in range(3):
	print("On random input " + str(r))
	inp = np.random.randn(2)
	print("Input is:", inp)
	print("--- --- --- ---")
	act_pattern = model.activation_pattern_from_input(inp)
	print("Act_pattern is:", act_pattern)
	print("--- --- --- ---")
	aff_params_all = model.affine_params(inp, flag='all')
	print("Affine params are:", aff_params_all)
	print("### ### ### ### ### ###")
'''