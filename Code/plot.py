import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

num_pts = 50

def list_colors(num_colors):
	''' Returns a list of matplotlib colors '''
	colors = []
	for name, hexa in matplotlib.colors.cnames.items():
		colors.append(name)
		if len(colors) >= num_colors:
			break
	# Debug : because 'antiquewhite' & 'blanchedalmond' look the same
	if num_colors == 16:
		return ['darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise']
	return colors

def map_to_cindex(actp, hidden_neurons):
	'''
	Maps a full activation pattern to an index in range(0,T)
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

	xps = np.linspace(min_val, max_val, num_pts)
	yps = np.linspace(min_val, max_val, num_pts)

	# Debug :: conclusion - 'antiquewhite'(1) & 'blanchedalmond'(8) look the same
	'''
	lis = []
	lis.append(np.array([-1.6, (0.25)*(-1.6) - 0.5 + 0.15]))
	lis.append(np.array([-2.4, (0.25)*(-2.4) - 0.5 - 0.15]))
	for pt in lis:
		act_pattern = model.activation_pattern_from_input(pt)
		print(act_pattern)
		color_ind = map_to_cindex(act_pattern, hidden_neurons)
		print(color_ind)
		print(colors[color_ind])
		plt.plot([pt[0]], [pt[1]], marker='x', markersize=6, color='k')
	'''
	# Plot model's honeycomb
	for x in xps:
		for y in yps:
			act_pattern = model.activation_pattern_from_input(np.array([x,y]))
			color_ind = map_to_cindex(act_pattern, hidden_neurons)
			plt.plot([x], [y], marker='o', markersize=4, color=colors[color_ind])
			# plt.text(x, y, str(color_ind), color="red", fontsize=3)

	# Plot the property separator
	for property_ in property_original:
		assert len(property_) == 1, "Can't handle disjunctions in plot"
		
		prop = property_[0]
		prop_comp = prop.split(' ')
		assert len(prop_comp) == 4, "Property doesn't have 4 components in plot"
		
		sign0 = (1 if prop_comp[0][0] == '+' else -1)
		logit0 = int(prop_comp[0][2:])
		sign1 = (1 if prop_comp[1][0] == '+' else -1)
		logit1 = int(prop_comp[1][2:])
		sign = prop_comp[2]
		value = float(prop_comp[3])
		
		labelled = False
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
						if not labelled:
							plt.plot([x], [y], marker='+', markersize=4, color='yellow', label='True separator of original property')
							labelled = True
						else:
							plt.plot([x], [y], marker='+', markersize=4, color='yellow')
					prev_ = curr_
	# Save figure
	plt.legend(loc='center', bbox_to_anchor=(0.5,1.1), ncol=2)
	plt.savefig(save_fname, dpi=600)
	# plt.clf()

# Helper function
def plot_linear_2d(w, b, min_val, max_val, color, label, labelled, delta=0):
	''' Plots w^Tx + b = delta in 2D '''
	assert w.shape == (2,), "Incorrect shape in plot_linear_2d"
	xps = np.linspace(min_val, max_val, num_pts)
	y = (-w[0]/w[1])*xps  + ((delta-b)/w[1])
	plt.ylim(min_val,max_val)
	if not labelled:
		plt.plot(xps, y, color+'-', label=label)
	else:
		plt.plot(xps, y, color+'-')

def plot_box_2d(box, color):
	x0_low, x0_high = box[0][0], box[0][1]
	x1_low, x1_high = box[1][0], box[1][1]
	rect = Rectangle((x0_low, x1_low), x0_high-x0_low, x1_high-x1_low, color=color, linestyle='-', linewidth=3)
	plt.gca().add_patch(rect)

def plot_act_pattern(model, point, act_pattern, weights, biases, save_fname, box):
	'''
	Plots point which is (supposed to be) the point that created act_pattern.
	Plots the given activation pattern also. Pattern can be relaxed.
	'''
	plt.plot([point[0]], [point[1]], marker='x', markersize=3, color='red')
	color = 'k'
	# Plot the activation pattern's linear boundaries
	# TODO : color the side of boundary that is in the pattern
	min_val, max_val = model.min_input_val, model.max_input_val
	hidlyrs, maxneurons = act_pattern.shape
	labelled = False
	for hidlyr in range(hidlyrs):
		for neuron in range(maxneurons):
			if act_pattern[hidlyr][neuron] in [0,1]:
				w = np.reshape(weights[hidlyr][neuron, :], (-1,))
				b = biases[hidlyr][neuron]
				plot_linear_2d(w, b, min_val, max_val, color, 'Paper\'s Activation Constraint', labelled)
				labelled = True
	# Save figure
	plot_box_2d(box, color)
	plt.legend(loc='center', bbox_to_anchor=(0.5,1.1), ncol=2)
	plt.savefig(save_fname, dpi=600)
	# plt.clf()

def plot_act_pattern_eps(model, point, act_pattern, weights, biases, save_fname, weight, bias, eps, box):
	plt.plot([point[0]], [point[1]], marker='x', markersize=3, color='red')
	# Plot the activation pattern's linear boundaries
	# TODO : color the side of boundary that is in the pattern
	color = 'b'
	min_val, max_val = model.min_input_val, model.max_input_val
	hidlyrs, maxneurons = act_pattern.shape
	labelled = False
	for hidlyr in range(hidlyrs):
		for neuron in range(maxneurons):
			if act_pattern[hidlyr][neuron] in [0,1]:
				w = np.reshape(weights[hidlyr][neuron, :], (-1,))
				b = biases[hidlyr][neuron]
				plot_linear_2d(w, b, min_val, max_val, color, 'Our Activation Constraint', labelled)
				labelled = True
	# Plot the linear boundary
	w = np.reshape(weight[0,:], (2,)) - np.reshape(weight[1,:], (2,))
	b = bias[0] - bias[1]
	plot_linear_2d(w, b, min_val, max_val, 'g', 'Our critical boundary', False, eps)
	# Draw the modified property line
	xps = np.linspace(min_val, max_val, num_pts)
	yps = np.linspace(min_val, max_val, num_pts)
	for x in xps:
			for y in yps:
				pt = np.array([x,y])
				logits = model.forward(pt)
				if logits[0] - logits[1] >= np.dot(w,pt) + b - eps:
					plt.plot([x], [y], marker='.', markersize=2, color='blanchedalmond')
	# Save figure
	plot_box_2d(box, color)
	plt.legend(loc='center', bbox_to_anchor=(0.5,1.1), ncol=2)
	plt.savefig(save_fname, dpi=600)
	# plt.clf()