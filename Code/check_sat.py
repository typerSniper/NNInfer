import sys, os
import subprocess
from subprocess import STDOUT, check_output
import math

PATH_TO_MARABOU = '../ML_Course/Marabou/build/Marabou'
TIMEOUT = 300

def print_constant(val):
	if val > 0:
		return "+" + str(val)
	else:
		return str(val)

def get_neuron_string(layer_id, neuron_id, val):
	upper_thresh = 0.0001
	neuron_name = "ws_" + str(layer_id+1) + "_" + str(neuron_id)
	value = ">= " + str(float(upper_thresh)) if val else "<= 0"
	return  ' '.join([neuron_name, value])


def get_activation_constraints(activation_pattern):
	activation_constraints = []
	for layer_id,layer in enumerate(activation_pattern):
		for neuron_id,val in enumerate(layer):
			if val == 0 or val == 1: 
				constraint = get_neuron_string(layer_id, neuron_id, val)
				activation_constraints.append(constraint)
	return activation_constraints

def write_to_file(constraints_string, file_name = "scratch/property.txt"):
	with open(file_name, 'w') as f:
		f.write(constraints_string)
	return file_name

def call_marabou(nnet_file, property_file):
	global PATH_TO_MARABOU
	global TIMEOUT
	flags = ['--verbosity=0']
	cmd = [PATH_TO_MARABOU, nnet_file, property_file] + flags
	is_timeout = 0
	try:
		out = check_output(cmd, stderr=STDOUT, timeout=TIMEOUT)
	except:
		is_timeout = 1
		print ("timeout")
		return True

	out = str(out)
	# print (out)
	# print(out.count('SAT'))
	assert out.count('SAT')==1, "Number of SAT in Marabou assumption failed"
	return (not "UNSAT" in out)

def make_dnf(cnf_properties):
	if len(cnf_properties)==0:
		return [[]]
	if len(cnf_properties)==1:
		dnf = []
		for literal in cnf_properties[0]:
			dnf.append([literal])
		return dnf
	dnf = []
	for clause in cnf_properties:
		assert (len(clause)==1), "assertion for cnf to dnf failed"
		dnf.append(clause[0])
	return [dnf]

def check_sat(activation_pattern, cnf_properties, nnet_file):
	print("checking sat")
	dnf_properties = make_dnf(cnf_properties)
	activation_pattern_list = get_activation_constraints(activation_pattern)
	for clause in dnf_properties:
		constraints = activation_pattern_list + clause
		constraints_string = '\n'.join(constraints)
		# print(constraints_string)
		property_file = write_to_file(constraints_string)
		sat = call_marabou(nnet_file, property_file)
		if sat:
			return sat
	return False

def negate_one_constraint(s):

	assert('<=' in s or '>=' in s)
	if '<=' in s:
		return s.replace('<=', '>=')
	else:
		return s.replace('>=', '<=')


def unfold_weight_vec(tup):
	len_inp= len(tup[1])
	weightM = tup[0]
	constr = ""
	for k in range(len_constr):
		constr+=" +(%d)x_%d"%(weightM[k], k)
	constr+=" <= %s"%(print_constant(tup[2])) #decide between <= or >=

	return constr

# def compare_left(nnet_file, sigma1_constraints, P, sigma2_constraints):
	
# 	property_constraints = ""
# 	for tup in P:
# 		property_constraints+=unfold_weight_vec(tup)

# 	lhs = property_constraints
# 	if len(sigma1_constraints)!=0:
# 		lhs+='\n'.join(sigma1_constraints)

# 	comp_ok = True
# 	for prop in sigma2_constraints:
# 		comp_ok = comp_ok and (not call_marabou(nnet_file, write_to_file(lhs+prop+'\n')) )
# 		if not comp_ok:
# 			break
	
# 	return comp_ok

# def compare_right(nnet_file, sigma1_constraints, P, sigma2_constraints):
	
# 	for tup in P:
# 		property_constraints.append(unfold_weight_vec(tup))
	
# 	lhs = ""
# 	if len(sigma2_constraints)!=0:
# 		lhs+='\n'.join(sigma2_constraints)

# 	comp_ok = True

# 	for prop in sigma1_constraints+property_constraints:
# 		comp_ok = comp_ok and (not call_marabou(nnet_file, write_to_file(lhs+prop+'\n')))
# 		if not comp_ok:
# 			break

# 	return comp_ok

 
	lhs_str = ""

	if len(lhs)!=0:
		lhs_str += '\n'.join(lhs)

	comp_ok = True

	for prop in rhs:
		neg_prop = negate_one_constraint(prop)
		comp_ok = comp_ok and (not call_marabou(nnet_file, write_to_file(lhs_str+neg_prop+'\n')))
		if not comp_ok:
			break
	
	return comp_ok

def compare_invariants(nnet_file, sigma1, P, epsilon, sigma2):

	property_constraints = []
	for tup in P:
		tup[2]+=epsilon # decide between + or -
		property_constraints.append(unfold_weight_vec(tup))

	sigma1_constraints = get_activation_constraints(sigma1)
	sigma2_constraints = get_activation_constraints(sigma2)

	lhs = sigma1_constraints+property_constraints
	rhs = sigma2_constraints

	if compare_left(nnet_file, lhs, rhs):
		print("sigma1 and P implies sigma2")
	elif compare_left(nnet_file, rhs, lhs):
		print("sigma2 implies sigma1 and P")
	else:
		print("orthogonal formulae")



