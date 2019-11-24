def get_negate_property_cnf(weights, bias, inp, pred_y, epsilon):
	def print_constant(val):
		if val > 0:
			return "+" + str(val)
		else:
			return str(val)

	def get_negate_property(pred_y, curr_y, diff_weights, diff_bias, epsilon):
		diff_y_term = "+y" + str(pred_y) + " -y" + str(curr_y)
		diff_weights_term = ' '.join([print_constant(-1*diff_weight) + "x" + str(idx) 
								for idx,diff_weight in enumerate(diff_weights)])
		value = "<="
		diff_bias_term = print_constant(diff_bias - epsilon)
		property_eqn = ' '.join([diff_y_term, diff_weights_term, value, diff_bias_term])
		return property_eqn

	num_out = weights.shape[0]
	clause = []
	for i in range(num_out):
		if i == pred_y:
			continue
		diff_weights = weights[pred_y] - weights[i]
		diff_bias = bias[pred_y] - bias[i]
		clause.append(get_negate_property(pred_y, i, diff_weights, diff_bias, epsilon))
	cnf = [clause]
	return cnf

