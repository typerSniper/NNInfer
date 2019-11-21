from check_sat import check_sat
from model import Model
import numpy as np

def main():
	num_neurons = [2,3,3,2]
	model = Model(num_neurons)
	nnet_name = 'dummy.nnet'
	model.write_NNET(nnet_name)
	for r in range(10):
		print("On random input " + str(r))
		inp = np.random.randn(2)
		activation_pattern = model.activation_pattern_from_input(inp)
		weights = model.affine_params(inp)
		properties = [['y0 >= y1']]
		is_sat = check_sat(nnet_name, activation_pattern, properties)
		print (is_sat)

if __name__ == '__main__':
    main()