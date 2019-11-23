import sys, os
import subprocess

PATH_TO_MARABOU = '../ML_Course/Marabou/build/Marabou'

def get_activation_constraints(activation_pattern):
	upper_thresh = 1e-8
	def get_constraint_string(layer_id, neuron_id, val):
		neuron_name = "ws_" + str(layer_id+1) + "_" + str(neuron_id)
		value = ">= " + str(int(upper_thresh)) if val else "<= 0"
		return  ' '.join([neuron_name, value])
	activation_constraints = []
	for layer_id,layer in enumerate(activation_pattern):
		for neuron_id,val in enumerate(layer):
			if val == 0 or val == 1: 
				constraint = get_constraint_string(layer_id, neuron_id, val)
				activation_constraints.append(constraint)
	return activation_constraints

def write_to_file(activation_pattern_list, conj_properties):
	file_name = "scratch/property.txt"
	with open(file_name, 'w') as f:
		constraints = activation_pattern_list + conj_properties
		constraints_string = '\n'.join(constraints)
		f.write(constraints_string)
	return file_name

def call_marabou(nnet_file, property_file):
	global PATH_TO_MARABOU
	flags = ['--verbosity=0']
	cmd = [PATH_TO_MARABOU, nnet_file, property_file] + flags
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	(out, err) = proc.communicate()
	out = str(out)
	# print(out)
	assert out.count('SAT')==1, "Number of SAT in Marabou assumption failed"
	return (not "UNSAT" in out)

def check_sat(activation_pattern, cnf_properties, nnet_file):
	print("checking sat")
	activation_pattern_list = get_activation_constraints(activation_pattern)
	for conj_properties in cnf_properties:
		property_file = write_to_file(activation_pattern_list, conj_properties)
		sat = call_marabou(nnet_file, property_file)
		if sat:
			return sat
	return False
