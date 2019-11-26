from check_sat import check_sat
from model import Model
import numpy as np
from minimal import findMinimal
from property import get_negate_property_cnf, get_negate_property_cnf_paper, get_property_cnf
from box import get_underapprox_box

def get_additional_constraints(weights_out, bias_out, pred_y):
	additional_constraints = []
	num_out = weights_out.shape[0]
	for i in range(num_out):
		if i==pred_y:
			continue
		diff_weights = weights_out[pred_y] - weights_out[i]
		diff_bias = bias_out[pred_y] - bias_out[i]
		additional_constraints.append((diff_weights, diff_bias))
	return additional_constraints


def run(inp, nnet_name, model, pred_y, epsilon, flag="our"):
	print (flag)
	activation_pattern = model.activation_pattern_from_input(inp)
	print ("the original activation pattern")
	print (activation_pattern)
	weights, bias = model.affine_params(inp, "all")
	if flag=='our':
		additional_constraints = get_additional_constraints(weights[-1], bias[-1], pred_y)
		property_negate = get_negate_property_cnf(inp, additional_constraints, pred_y, epsilon)
	else:
		additional_constraints = []
		property_negate = get_negate_property_cnf_paper(pred_y, model.num_neurons[-1])
	print (property_negate)
	is_sat = check_sat(activation_pattern, property_negate, nnet_name)
	if not is_sat:
		activation_pattern = findMinimal(activation_pattern, property_negate, nnet_name)
		assert(not check_sat(activation_pattern, property_negate, nnet_name))
		print ("minimalized activation pattern")
		print (activation_pattern)
	else:
		print ("property not satisfied")
		additional_constraints = get_additional_constraints(weights[-1], bias[-1], pred_y)
		epsilon = 0
	under_box = get_underapprox_box(activation_pattern, weights, bias, model.min_input_val, model.max_input_val, additional_constraints, epsilon)
	print (under_box)
	print ("########################")




def main():
	num_neurons = [2,4,2]
	epsilon = 0.0001
	nnet_name = 'scratch/dummy.nnet'
	min_val = -3
	max_val = 3
	pred_y = 0
	weights_init = {}
	weights_init[1] = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, 2.0], [1.0, -4.0]])
	weights_init[2] = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
	biases_init = {}
	biases_init[1] = np.array([1.0, 0.0, -2.0, -2.0])
	biases_init[2] = np.zeros(2)
	model = Model(num_neurons, weights_init=weights_init, biases_init=biases_init, min_input_val=min_val, max_input_val=max_val)
	model.write_NNET(nnet_name)
	for r in range(1):
		inp = np.random.randn(num_neurons[0])
		print ("on input")
		print (inp)
		print ("original property")
		print (get_property_cnf(pred_y, num_neurons[-1]))
		run(inp, nnet_name, model, pred_y, epsilon, "our")
		run(inp, nnet_name, model, pred_y, epsilon, "paper")


if __name__ == '__main__':
    main()