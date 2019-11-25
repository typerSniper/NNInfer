from check_sat import check_sat
from model import Model
import numpy as np
from minimal import findMinimal
from property import get_negate_property_cnf
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



def main():
	num_neurons = [2,3,3,2]
	epsilon = 1e-4
	nnet_name = 'scratch/dummy.nnet'
	min_val = -3
	max_val = 3
	pred_y = 0
	model = Model(num_neurons, min_input_val=min_val, max_input_val=max_val)
	model.write_NNET(nnet_name)
	for r in range(1):
		print("On random input " + str(r))
		inp = np.random.randn(2)

		
		activation_pattern = model.activation_pattern_from_input(inp)
		weights, bias = model.affine_params(inp, "all")
		additional_constraints = get_additional_constraints(weights[-1], bias[-1], pred_y)
		property_our = get_negate_property_cnf(inp, additional_constraints, pred_y, epsilon)
		is_sat = check_sat(activation_pattern, property_our, nnet_name)
		if not is_sat:
			activation_pattern = findMinimal(activation_pattern, property_our, nnet_name)
			assert(not check_sat(activation_pattern, property_our, nnet_name))
			print("here")
		print (activation_pattern)
		under_box = get_underapprox_box(activation_pattern, weights, bias, min_val, max_val, additional_constraints, epsilon)
		print (under_box)
		print ("########################")


		activation_pattern = model.activation_pattern_from_input(inp)
		weights, bias = model.affine_params(inp, "all")
		additional_constraints = []
		property_paper = [['+y0 -y1 <= 0']]
		is_sat = check_sat(activation_pattern, property_paper, nnet_name)
		if not is_sat:
			activation_pattern = findMinimal(activation_pattern, property_paper, nnet_name)
			assert(not check_sat(activation_pattern, property_paper, nnet_name))
			print("here")
		print (activation_pattern)
		under_box = get_underapprox_box(activation_pattern, weights, bias, min_val, max_val, additional_constraints, epsilon)
		print (under_box)
		print ("########################")

		


if __name__ == '__main__':
    main()