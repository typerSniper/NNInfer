from check_sat import check_sat
from model import Model
import numpy as np
from minimal import findMinimal
from property import get_negate_property_cnf
from box import get_underapprox_box

def main():
	num_neurons = [2,3,3,2]
	epsilon = 0.01
	model = Model(num_neurons)
	nnet_name = 'scratch/dummy.nnet'
	model.write_NNET(nnet_name)
	for r in range(1):
		print("On random input " + str(r))
		inp = np.random.randn(2)
		activation_pattern = model.activation_pattern_from_input(inp)

		weights, bias = model.affine_params(inp)
		property_our = get_negate_property_cnf(weights, bias, inp, 0, epsilon)
		is_sat = check_sat(activation_pattern, property_our, nnet_name)
		if not is_sat:
			activation_pattern = findMinimal(activation_pattern, property_our, nnet_name)
			assert(not check_sat(activation_pattern, property_our, nnet_name))
			print("here")
		# print (is_sat)
		print (activation_pattern)
		weights, bias = model.affine_params(inp, 'all')
		under_box = get_underapprox_box(activation_pattern, weights, bias, [-3]*2, [3]*2, [], epsilon)
		print (under_box)
		
		
		print ("########################")
		property_paper = [['+y0 -y1 <= 0']]
		activation_pattern = model.activation_pattern_from_input(inp)
		is_sat = check_sat(activation_pattern, property_paper, nnet_name)
		if not is_sat:
			activation_pattern = findMinimal(activation_pattern, property_paper, nnet_name)
			assert(not check_sat(activation_pattern, property_paper, nnet_name))
			print("here")
		# print (is_sat)
		print (activation_pattern)
		print ("########################")

		


if __name__ == '__main__':
    main()