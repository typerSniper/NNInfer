from check_sat import print_constant

def get_negate_property_cnf(inp, additional_constraints, pred_y, epsilon):
	def get_negate_property(pred_y, curr_y, diff_weights, diff_bias, epsilon):
		diff_y_term = "+y" + str(pred_y) + " -y" + str(curr_y)
		diff_weights_term = ' '.join([print_constant(-1*diff_weight) + "x" + str(idx) 
								for idx,diff_weight in enumerate(diff_weights)])
		value = "<="
		diff_bias_term = print_constant(diff_bias - epsilon)
		property_eqn = ' '.join([diff_y_term, diff_weights_term, value, diff_bias_term])
		return property_eqn

	num_out = len(additional_constraints) + 1
	idx = 0
	clause = []
	for i in range(num_out):
		if i == pred_y:
			continue
		diff_weights, diff_bias = additional_constraints[idx]
		clause.append(get_negate_property(pred_y, i, diff_weights, diff_bias, epsilon))
		idx+=1
	cnf = [clause]
	return cnf

