def get_negate_property_cnf(weights, bias, inp, pred_y, epsilon):
	def get_negate_property(pred_y, curr_y, diff_weights, diff_bias, epsilon):
		diff_y_term = "+y" + str(pred_y) + " -y" + str(curr_y)
		diff_weights_term = ' '.join([str(-1*diff_weight) + "x" + str(idx) 
								for idx,diff_weight in enumerate(diff_weights)])
		diff_bias_term = str(-1*diff_bias)
		epsilon_term = "+" + str(epsilon)
		value = "<= 0"
		property_eqn = ' '.join([diff_y_term, diff_weights_term, diff_bias_term, epsilon_term, value])
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

