import sys, os

def get_activation_constraints(activation_pattern):
	upper_thresh = 1e-8
	def get_constraint_string(layer_id, neuron_id, val):
		neuron_name = "ws_" + str(layer_id+1) + "_" + str(neuron_id)
		value = ">=" + str(upper_thresh) if val else "<=0"
		return  ' '.join(neuron_name, value)
	activation_constraints = []
	for layer_id,layer in enumerate(activation_pattern):
		for neuron_id,val in enumerate(layer):
			if val == 0 or val == 1: 
				constraint = get_constraint_string(layer_id, neuron_id, val)
				activation_constraints.append(constraint)
	return '\n'.join(activation_constraints)

def get_property_constraints(property_string_list):
	return '\n'.join(property_string_list)



