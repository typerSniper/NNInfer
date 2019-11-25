import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def list_colors(num_colors):
	''' Returns a list of matplotlib colors '''
	colors = []
	for name, hexa in matplotlib.colors.cnames.items():
		colors.append(name)
		if len(colors) >= num_colors:
			break
	return colors

def map_to_cindex(actp, hidden_neurons):
	'''
	Maps an activation pattern to an index in range(0,T)
	T = total number of activation patterns possible
	'''
	actp_vec = np.reshape(actp, (-1,))
	actp_trunc = actp_vec[actp_vec != 2.0]
	assert actp_trunc.shape == (hidden_neurons,), "act_pattern should be complete in plot"
	mult_arr = np.array([2**i for i in range(hidden_neurons)])
	return int(np.sum(actp_trunc*mult_arr))

def plot_base(model, property_original, save_fname):
	'''
	Base plot of the underlying honeycomb structure
	and the original property that we want.
	'''
	min_val, max_val = model.min_input_val, model.max_input_val
	hidden_neurons = model.hidden_neurons
	colors = list_colors(2**hidden_neurons)

	xps = np.linspace(min_val, max_val, 100)
	yps = np.linspace(min_val, max_val, 100)

	# Plot model's honeycomb
	for x in xps:
		for y in yps:
			act_pattern = model.activation_pattern_from_input(np.array([x,y]))
			color_ind = map_to_cindex(act_pattern, hidden_neurons)
			plt.plot([x], [y], marker='o', markersize=2, color=colors[color_ind])

	# Plot the property separator
	for property_ in property_original:
		assert len(property_) == 1, "Can't handle disjunctions in plot"
		
		prop = property_[0]
		prop_comp = prop.split(' ')
		assert len(prop_comp) == 4, "Property doesn't have 4 components in plot"
		
		sign0 = 1 if prop_comp[0][0] == '+' else -1
		logit0 = int(prop_comp[0][2:])
		sign1 = 1 if prop_comp[1][0] == '+' else -1
		logit1 = int(prop_comp[1][2:])
		sign = prop_comp[2]
		value = float(prop_comp[3])
		
		for x in xps:
			prev_ = np.inf
			for y in yps:
				logits = model.forward(np.array([x,y]))
				curr_ = (sign0*logits[logit0] + sign1*logits[logit1] <= value)
				if sign != '<=':
					curr_ = (sign0*logits[logit0] + sign1*logits[logit1] >= value)
				if y == min_val:
					prev_ = curr_
				else:
					if curr_ != prev_:
						plt.plot([x], [y], marker='+', markersize=3, color='yellow')
	# Save figure
	plt.savefig(save_fname)
	plt.clf()

# Helper function
def plot_linear_2d(w, b, min_val, max_val, delta=0):
	''' Plots w^Tx + b = delta in 2D '''
	assert w.shape == (2,), "Incorrect shape in plot_linear_2d"
	xps = np.linspace(min_val, max_val, 100)
	y = (-w[0]/w[1])*xps  + ((delta-b)/w[1])
	plt.plot(xps, y, 'k-')

def plot_act_pattern(model, point, act_pattern, weights, biases, save_fname):
	'''
	Plots point which is (supposed to be) the point that created act_pattern.
	Plots the given activation pattern also. Pattern can be relaxed.
	'''
	plt.plot([point[0]], [point[1]], marker='x', markersize=3, color='red')
	# Plot the activation pattern's linear boundaries
	# TODO : color the side of boundary that is in the pattern
	min_val, max_val = model.min_input_val, model.max_input_val
	hidlyrs, maxneurons = act_pattern.shape
	for hidlyr in range(hidlyrs):
		for neuron in range(maxneurons):
			if act_pattern[hidlyr][neuron] in [0,1]:
				w = np.reshape(weights[hidlyr][neuron, :], (-1,))
				b = biases[hidlyr][neuron]
				plot_linear_2d(w, b, min_val, max_val)
	# Save figure
	plt.savefig(save_fname)
	plt.clf()

def plot_act_pattern_eps(model, point, act_pattern, weights, biases, save_fname, weight, bias, eps):
	plt.plot([point[0]], [point[1]], marker='x', markersize=3, color='red')
	# Plot the activation pattern's linear boundaries
	# TODO : color the side of boundary that is in the pattern
	min_val, max_val = model.min_input_val, model.max_input_val
	hidlyrs, maxneurons = act_pattern.shape
	for hidlyr in range(hidlyrs):
		for neuron in range(maxneurons):
			if act_pattern[hidlyr][neuron] in [0,1]:
				w = np.reshape(weights[hidlyr][neuron, :], (-1,))
				b = biases[hidlyr][neuron]
				plot_linear_2d(w, b, min_val, max_val)
	# Plot the linear boundary
	w = np.reshape(weights[0,:], (2,)) - np.reshape(weights[1,:], (2,))
	b = biases[0] - biases[1]
	plot_linear_2d(w, b, min_val, max_val, eps)
	# Save figure
	plt.savefig(save_fname)
	plt.clf()