from readNNet import readNNet
from check_sat import check_sat
from model import Model
import numpy as np
from minimal import findMinimal
from property import get_negate_property_cnf, get_negate_property_cnf_paper, get_property_cnf
from box import get_underapprox_box, get_underapprox_box_vol
from plot import *
from read_mnist import get_best_inputs_mnist, get_support
from time import time

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


def run(inp, nnet_name, model, pred_y, epsilon, plot_flag, fp, flag="our"):
	fp.write("##evaluating" + str(flag))
	fp.write("##epsilon" + str(epsilon))
	print (flag)
	activation_pattern = model.activation_pattern_from_input(inp)
	fp.write("##activation_pattern_org" + str(activation_pattern))
	print ("the original activation pattern")
	print (activation_pattern)
	weights, bias = model.affine_params(inp, "all")
	if flag=='our':
		additional_constraints = get_additional_constraints(weights[-1], bias[-1], pred_y)
		property_negate = get_negate_property_cnf(inp, additional_constraints, pred_y, epsilon)
	else:
		additional_constraints = []
		property_negate = get_negate_property_cnf_paper(pred_y, model.num_neurons[-1])
	fp.write("##property_negate" + str(property_negate))
	# print (property_negate)
	is_sat = check_sat(activation_pattern, property_negate, nnet_name)
	if not is_sat:
		activation_pattern = findMinimal(activation_pattern, property_negate, nnet_name)
		# assert(not check_sat(activation_pattern, property_negate, nnet_name))
		print ("minimalized activation pattern")
		fp.write("##minimalized_activation_pattern" + str(activation_pattern))
		print (activation_pattern)
	else:
		print ("property not satisfied")
		additional_constraints = get_additional_constraints(weights[-1], bias[-1], pred_y)
		fp.write("##property_not_satisfied" + str(activation_pattern))
		epsilon = 0
	under_box_sum, vol_sum, peri = get_underapprox_box(activation_pattern, weights, bias, model.min_input_val, model.max_input_val, additional_constraints, epsilon)
	under_box_vol, vol_vol = get_underapprox_box_vol(activation_pattern, weights, bias, model.min_input_val, model.max_input_val, additional_constraints, epsilon)
	fp.write("##under_box_sum" + str(under_box_sum))
	fp.write("##under_box_sum perimeter" + str(peri))
	fp.write("##under_box_sum volume" + str(vol_sum))
	fp.write("##under_box_vol" + str(under_box_vol))
	fp.write("##under_box_vol log_volume" + str(vol_vol))
	support_train = get_support(activation_pattern, model, additional_constraints, epsilon, inp)
	fp.write("##support in train" + str(support_train))
	fp.write("##end##")
	print("vol log_volume" + str(vol_vol))
	print("sum perimeter" + str(peri))
	print("support in train" + str(support_train))
	if plot_flag:
		if flag == 'our':
			plot_act_pattern_eps(model, inp, activation_pattern, weights, bias, 
				'scratch_mnist/our'+str(epsilon)+'.png', weights[-1], bias[-1], epsilon, under_box_vol)
		else:
			plot_act_pattern(model, inp, activation_pattern, weights, bias, 'scratch_mnist/paper.png', under_box_vol)
	# print ("########################")




def main():
	nnet_name = 'data/anwu_mnist_10_layer.txt'
	num_inputs = 50
	# epsilon_list = [0.01]
	epsilon_list = [0.00001, 0.001, 0.1, 1]
	plot_flag= 0
	
	wlis, blis, mins, maxes, _, _ = readNNet(nnet_name, withNorm=True)
	assert len(wlis) == len(blis)
	lyrs = len(wlis)+1

	min_val = min(mins)
	max_val = max(maxes)
	# Create num_neurons, weights_init, biases_init
	weights_init = {}
	biases_init = {}
	num_neurons = [wlis[0].shape[1], wlis[0].shape[0]]
	for lyr in range(lyrs-1):
		weights_init[lyr+1] = wlis[lyr]
		biases_init[lyr+1] = blis[lyr]
		if lyr > 0:
			num_neurons.append(wlis[lyr].shape[0])

	model = Model(num_neurons, weights_init=weights_init, biases_init=biases_init, min_input_val=min_val, max_input_val=max_val)
	mnist_points = get_best_inputs_mnist(model, num_inputs)
	for idx, inp_tuple in enumerate(mnist_points):
		inp, pred_y = inp_tuple
		pred_y = pred_y[0]
		file_name = "scratch_mnist/mnist_" + str(idx) + ".out"
		fp = open(file_name,'w')
		start_time = time()
		fp.write("##input" + str(inp))
		fp.write("##pred_y" + str(inp))
		print ("on input")
		print (inp)
		print ("file" + file_name)
		print ("original property")
		property_orig = get_property_cnf(pred_y, num_neurons[-1])
		fp.write("##property-to-prove" + str(property_orig))
		print (property_orig)
		if plot_flag:
			plot_base(model, property_orig, 'scratch_mnist/base.png')

		run(inp, nnet_name, model, pred_y, -1, plot_flag, fp, "paper")
		for epsilon in epsilon_list:
			run(inp, nnet_name, model, pred_y, epsilon, plot_flag, fp, "our")
		
		print ("time on this inp" + str(time() - start_time))
		print ("######################")
		fp.close()
if __name__ == '__main__':
    main()