import sys, os
import subprocess


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
	return activation_constraints

def write_to_file(activation_pattern_string, conj_properties):
	file_name = "scratch/property.txt"
	with open(file_name) as f:
		constraints = activation_pattern_string + '\n' + conj_properties
		f.write(constraints)
	return file_name

def call_marabou(path_to_marabou, nnet_file, property_file):
	flags = ['--verbosity=0']
	cmd = [path_to_marabou, nnet_file, property_file, flags]
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
	(out, err) = proc.communicate()
	assert out.count('SAT')==1, "Number of SAT in Marabou assumption failed"
	return (not "UNSAT" in out)

def check_sat(path_to_marabou, nnet_file, activation_pattern, cnf_properties):
	activation_pattern_string = get_activation_constraints(activation_pattern)
	for conj_properties in cnf_properties:
		property_file = write_to_file(activation_pattern_string, conj_properties)
		sat = call_marabou(path_to_marabou, nnet_file, property_file)
		if sat:
			return sat
	return False