import pulp
import numpy as np

def get_underapprox_box(activation_pattern, weights, biases, min_val, max_val, additional_constraints, epsilon):
	def get_w_t_x(weight_vector, pulpInputs):
		w_t_hi = [weight_vector[i] * pulpInputs[i][1] for i in range(num_inputs)]
		w_t_lo = [weight_vector[i] * pulpInputs[i][0] for i in range(num_inputs)]
		return w_t_hi, w_t_lo

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
	for layer_id,layer in enumerate(activation_pattern):
		weight_matrix = weights[layer_id]
		bias_vector = biases[layer_id]
		for neuron_id,neuron in enumerate(layer):
			if neuron==2:
				continue
			weight_vector = weight_matrix[neuron_id]
			b = bias_vector[neuron_id]
			w_t_hi, w_t_lo = get_w_t_x(weight_vector, pulpInputs)
			constraint_name = "activation constraint for neuron" + str(layer_id) + "_" + str(neuron_id)
			constraint_name_hi = constraint_name + " high"
			constraint_name_lo = constraint_name + " low"
			if neuron==0:
				prob += pulp.lpSum(w_t_hi) <= -1 * b, constraint_name_hi
				prob += pulp.lpSum(w_t_lo) <= -1 * b, constraint_name_lo
			elif neuron==1:
				prob += pulp.lpSum(w_t_hi) >= -1 * b, constraint_name_hi
				prob += pulp.lpSum(w_t_lo) >= -1 * b, constraint_name_lo
	for constraint_id, constraint in enumerate(additional_constraints):
		diff_weight, diff_bias = constraint
		w_t_hi, w_t_lo = get_w_t_x(diff_weight, pulpInputs)
		constraint_name = "additinal property constraint " + str(constraint_id)
		constraint_name_hi = constraint_name + " high"
		constraint_name_lo = constraint_name + " low"
		rhs = epsilon - diff_bias
		prob += pulp.lpSum(w_t_hi) >= rhs, constraint_name_hi
		prob += pulp.lpSum(w_t_lo) >= rhs, constraint_name_lo
	# print (prob)
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





