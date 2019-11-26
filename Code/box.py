import pulp
import numpy as np

UPPER_THRESH = 1e-8

def get_underapprox_box(activation_pattern, weights, biases, min_val, max_val, additional_constraints, epsilon):
	def get_w_t_x(weight_vector, pulpInputs):
		w_t_x = []
		num_inputs = len(weight_vector)
		for i in range(num_inputs):
			coeff = weight_vector[i]
			if coeff >= 0:
				w_t_x.append(coeff * pulpInputs[i][1])
			else:
				w_t_x.append(coeff * pulpInputs[i][0])
		return w_t_x

	num_inputs = weights[0].shape[1]
	pulpInputs = []
	for i in range(num_inputs):
		var_name = 'x' + str(i)
		hi = var_name + '_hi'
		lo = var_name + '_low'
		d_hi = pulp.LpVariable(hi, lowBound=min_val, upBound=max_val, cat='Continuous') 
		d_lo = pulp.LpVariable(lo, lowBound=min_val, upBound=max_val, cat='Continuous')
		pulpInputs.append((d_lo, d_hi))
	prob = pulp.LpProblem("Box", pulp.LpMaximize)
	prob += pulp.lpSum([(pulpInputs[i][1]-pulpInputs[i][0]) for i in range(num_inputs)]), "Total Range is Maximized"
	for i in range(num_inputs):
		prob += (pulpInputs[i][1]-pulpInputs[i][0]) >= 0, "High > Low Constraint for x_" + str(i) 
	for layer_id,layer in enumerate(activation_pattern):
		weight_matrix = weights[layer_id]
		bias_vector = biases[layer_id]
		for neuron_id,neuron in enumerate(layer):
			if neuron==2:
				continue
			weight_vector = weight_matrix[neuron_id]
			b = bias_vector[neuron_id]
			constraint_name = "activation constraint for neuron" + str(layer_id) + "_" + str(neuron_id)
			if neuron==0:
				w_t_x = get_w_t_x(weight_vector, pulpInputs)
				prob += pulp.lpSum(w_t_x) <= -1 * b, constraint_name
			elif neuron==1:
				w_t_x = get_w_t_x(-1*weight_vector, pulpInputs)
				prob += pulp.lpSum(w_t_x) <= (b - UPPER_THRESH), constraint_name
	for constraint_id, constraint in enumerate(additional_constraints):
		diff_weight, diff_bias = constraint
		w_t_x = get_w_t_x(-1*diff_weight, pulpInputs)
		constraint_name = "additinal property constraint " + str(constraint_id)
		rhs = -1*epsilon + diff_bias - UPPER_THRESH
		prob += pulp.lpSum(w_t_x) <= rhs, constraint_name
	print (prob)
	status = prob.solve()   
	result = pulp.LpStatus[status]
	print (result)
	under_approx_box = []
	if result == 'Infeasible':
		for i in range(num_inputs):
			under_approx_box.append((0, 0))
		return under_approx_box	
	assert result=='Optimal', "The under-approximate constraint problem fails to give optimal solution"
	for i in range(num_inputs):
		under_approx_box.append((pulp.value(pulpInputs[i][0]), pulp.value(pulpInputs[i][1])))
	return under_approx_box





