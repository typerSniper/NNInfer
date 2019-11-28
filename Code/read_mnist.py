import numpy as np
from model import Model
from readNNet import readNNet
import json


Xtrain = np.load('data/train_x.npy')
Ytrain = np.load('data/train_y.npy')
print (Xtrain.shape)

actp_dict = dict()

def get_best_inputs_mnist(model, num_inputs):
	for idx,inp in enumerate(Xtrain):
		actp = model.activation_pattern_from_input(inp)
		actp_str = str(actp)
		if not actp_str in actp_dict:
			actp_dict[actp_str] = []
		actp_dict[actp_str].append(((inp/255.0), Ytrain[idx]))

	cnt = 0
	fg = []
	for key in list(actp_dict.keys()):
		val = actp_dict[key]
		cnt += len(val)
		fg.append((len(val), val[0]))
	fg = sorted(fg, key=lambda tup: tup[0])
	fg.reverse()
	inps = []
	for i in range(num_inputs):
		inps.append(fg[i][1])
	return inps


def subset_act(activation_pattern, inp_activation_pattern):
	# print(activation_pattern)
	# print(inp_activation_pattern)
	for layer_idx, layer_act  in enumerate(activation_pattern):
		for neuron_idx, neuron_val in enumerate(layer_act):
			if neuron_val != 2.0 and neuron_val != inp_activation_pattern[layer_idx,neuron_idx]:
				return False
	return True

def check_additional(additional_constraints, epsilon, inp):
	def get_lhs(weight_vector, inp):
		summ = 0
		num_inputs = len(weight_vector)
		for i in range(num_inputs):
			coeff = weight_vector[i]
			summ += (coeff * inp[i])
		return summ
	for wt_tuple in additional_constraints:
		diff_weight, diff_bias = wt_tuple
		lhs = get_lhs(-1*diff_weight, inp)
		rhs = -1*epsilon + diff_bias 
		if lhs >= rhs:
			return False
	return True


def get_support(activation_pattern, model, additional_constraints, epsilon, curr_inp):
	support = 0
	for inp in Xtrain:
		inp = inp/255.0
		inp_activation_pattern = model.activation_pattern_from_input(inp)
		if subset_act(activation_pattern, inp_activation_pattern):
			if check_additional(additional_constraints, epsilon, inp):
				support +=1
	return support