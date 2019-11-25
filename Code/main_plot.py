from check_sat import check_sat
from model import Model
import numpy as np
from minimal import findMinimal
from property import get_negate_property_cnf
from plot import *

num_neurons = [2,4,2]
weights_init = {}
weights_init[1] = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, 2.0], [1.0, -4.0]])
weights_init[2] = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
biases_init = {}
biases_init[1] = np.array([1.0, 0.0, -2.0, -2.0])
biases_init[2] = np.zeros(2)
model = Model(num_neurons, weights_init, biases_init)
nnet_name = 'scratch/dummy_init_plot.nnet'
model.write_NNET(nnet_name)

property_orig = [['+y0 -y1 >= 0']]
negate_property_paper = [['+y0 -y1 <= 0']]
plot_base(model, property_orig, 'scratch/base.png')

inp = np.array([1.0, 0.0])
weights_all, biases_all = model.affine_params(inp, flag='all')
weights, biases = model.affine_params(inp)

# Paper's approach
activation_pattern = model.activation_pattern_from_input(inp)
is_sat = check_sat(activation_pattern, negate_property_paper, nnet_name)
if not is_sat:
	activation_pattern = findMinimal(activation_pattern, negate_property_paper, nnet_name)
	assert(not check_sat(activation_pattern, negate_property_paper, nnet_name))
plot_act_pattern(model, inp, activation_pattern, weights_all, biases_all, 'scratch/paper.png')
print ("########################")	

# Our approach
eps_list = [0.01, 0.1, 1.0, 10.0]
for eps in eps_list:
	activation_pattern = model.activation_pattern_from_input(inp)
	property_our = get_negate_property_cnf(weights, bias, inp, 0, eps)
	is_sat = check_sat(activation_pattern, property_our, nnet_name)
	if not is_sat:
		activation_pattern = findMinimal(activation_pattern, property_our, nnet_name)
		assert(not check_sat(activation_pattern, property_our, nnet_name))
	plot_act_pattern_eps(model, inp, activation_pattern, weights_all, biases_all, 
		'scratch/our'+str(eps)+'.png', weights, biases, eps)
print ("########################")