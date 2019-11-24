import pulp
import numpy as np

def get_underapprox_box(activation_pattern, weights, biases, diff_weights, diff_biases, epsilon):
	num_inputs = weights[0].ahape[1]