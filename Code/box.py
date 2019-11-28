import pulp
import numpy as np
import cvxpy as cp

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
	# print (prob)
	status = prob.solve()   
	result = pulp.LpStatus[status]
	print (result)
	under_approx_box = []
	if result == 'Infeasible':
		for i in range(num_inputs):
			under_approx_box.append((0, 0))
		return under_approx_box, 0, 0
	assert result=='Optimal', "The under-approximate constraint problem fails to give optimal solution"
	for i in range(num_inputs):
		under_approx_box.append((pulp.value(pulpInputs[i][0]), pulp.value(pulpInputs[i][1])))
	perimeter = 0.0
	volume = 1.0
	for i in range(num_inputs):
		volume *= (pulp.value(pulpInputs[i][1]) - pulp.value(pulpInputs[i][0]))
		perimeter += pulp.value(pulpInputs[i][1]) - pulp.value(pulpInputs[i][0])
	return under_approx_box, volume, perimeter


def get_underapprox_box_vol(activation_pattern, weights, biases, min_val, max_val, additional_constraints, epsilon):
	def get_w_t_x(weight_vector):
		w_t_x = []
		num_inputs = len(weight_vector)
		for i in range(num_inputs):
			coeff = weight_vector[i]
			if coeff >= 0:
				w_t_x.append(coeff * var_high[i])
			else:
				w_t_x.append(coeff * var_low[i])
		return w_t_x

	num_inputs = weights[0].shape[1]
	var_low = cp.Variable(num_inputs)
	var_high = cp.Variable(num_inputs)
	constraints = []
	for i in range(num_inputs):
		d_low = var_low[i]
		d_high = var_high[i]
		constraints.append(d_low >= min_val)
		constraints.append(d_high <= max_val)
		constraints.append(d_high - d_low >= 0)
	for layer_id,layer in enumerate(activation_pattern):
		weight_matrix = weights[layer_id]
		bias_vector = biases[layer_id]
		for neuron_id,neuron in enumerate(layer):
			if neuron==2:
				continue
			weight_vector = weight_matrix[neuron_id]
			b = bias_vector[neuron_id]
			if neuron==0:
				w_t_x = get_w_t_x(weight_vector)
				constraints.append(cp.sum(w_t_x) <= -1 * b)
			elif neuron==1:
				w_t_x = get_w_t_x(-1*weight_vector)
				constraints.append(cp.sum(w_t_x) <= (b - UPPER_THRESH))
	for constraint_id, constraint in enumerate(additional_constraints):
		diff_weight, diff_bias = constraint
		w_t_x = get_w_t_x(-1*diff_weight)
		rhs = -1*epsilon + diff_bias - UPPER_THRESH
		constraints.append(cp.sum(w_t_x) <= rhs)
	obj = cp.Maximize(cp.sum(cp.log(var_high - var_low)))
	prob = cp.Problem(obj, constraints)
	# print (prob)
	prob.solve(solver='ECOS')
	result = prob.status
	print (result)
	under_approx_box = []
	if result == cp.INFEASIBLE:
		for i in range(num_inputs):
			under_approx_box.append((0, 0))
		return under_approx_box, -np.inf
	assert result==cp.OPTIMAL, "The under-approximate constraint problem fails to give optimal solution"
	sol_low = var_low.value
	sol_high = var_high.value
	for i in range(num_inputs):
		under_approx_box.append((sol_low[i], sol_high[i]))
	log_volume = np.sum(np.log(sol_high - sol_low + 1e-8))
	return under_approx_box, log_volume


