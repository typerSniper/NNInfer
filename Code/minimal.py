import numpy as np


# asssumes that activation_pattern implies properties
def findMinimal (activation_pattern, properties):
	assert(check_sat(activation_pattern, properties, nnet))

	num_layers = activation_pattern.shape

	m_activation_pattern = np.copy(activation_pattern)

	for layer in range(num_layers[0]-1, -1, -1):
		
		num_neurons = (m_activation_pattern[layer].shape)[0]
		m_activation_pattern[layer] = np.repeat(2, num_neurons)
		
		if check_sat(m_activation_pattern, properties, nnet):
			continue
		else:
			m_activation_pattern[layer] = activation_pattern[layer]
			return findMinimalInLayer(m_activation_pattern, properties, layer)
	
	print("Empty activation_pattern!")
	return m_activation_pattern


def getNext(candidates, activation_pattern, properties, layer):
	return np.amin(candidates)

def findMinimalInLayer(activation_pattern, properties, layerNum):
	num_neurons = (activation_pattern.shape)[0]
	candidates = np.arange(num_neurons)
	# activation_pattern = np.copy(activation_pattern)
	while len(candidates)!=0:

		candidate = getNext(candidates, activation_pattern, properties, layerNum)
		np.delete(candidates, candidate)
		initial_status = activation_pattern[layer][candidate]
		activation_pattern[layer][candidate] = 2

		if initial_status==2 or check_sat(activation_pattern, properties, nnet):
			continue
		else:
			activation_pattern[layer][candidate] = initial_status
	return activation_pattern			



