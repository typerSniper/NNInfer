from check_sat import check_sat
from model import Model
import numpy as np
from minimal import findMinimal
from property import get_negate_property_cnf

def main():
	num_neurons = [2,3,3,2]
	model = Model(num_neurons)
	nnet_name = 'scratch/dummy.nnet'
	model.write_NNET(nnet_name)
	for r in range(1):
		print("On random input " + str(r))
		inp = np.random.randn(2)
		activation_pattern = model.activation_pattern_from_input(inp)

		weights, bias = model.affine_params(inp)
		property_our = get_negate_property_cnf(weights, bias, inp, 0, 0)
		is_sat = check_sat(activation_pattern, property_our, nnet_name)
		if not is_sat:
			activation_pattern = findMinimal(activation_pattern, property_our, nnet_name)
			assert(not check_sat(activation_pattern, property_our, nnet_name))
			print("here")
		# print (is_sat)
		print (activation_pattern)
		
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