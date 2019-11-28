import numpy as np

from check_sat import check_sat

# asssumes that activation_pattern implies properties
def findMinimal (activation_pattern, properties, nnet):
	# assert(not check_sat(activation_pattern, properties, nnet))

	num_layers = activation_pattern.shape

	m_activation_pattern = np.copy(activation_pattern)

	for layer in range(num_layers[0]-1, -1, -1):
		
		num_neurons = (m_activation_pattern[layer].shape)[0]
		m_activation_pattern[layer] = np.repeat(2, num_neurons)
		if not check_sat(m_activation_pattern, properties, nnet):
			continue
		else:
			m_activation_pattern[layer] = activation_pattern[layer]
			return findMinimalInLayer(m_activation_pattern, properties, layer, nnet)
	return m_activation_pattern


def getNext(candidates, activation_pattern, properties, layer):
	return np.amin(candidates)

def findMinimalInLayer(activation_pattern, properties, layerNum, nnet):
	num_neurons = (activation_pattern[layerNum].shape)[0]
	candidates = np.arange(num_neurons)
	# activation_pattern = np.copy(activation_pattern)
	while len(candidates)!=0:

		candidate = getNext(candidates, activation_pattern, properties, layerNum)
		candidates = candidates[candidates != candidate]
		initial_status = activation_pattern[layerNum][candidate]
		activation_pattern[layerNum][candidate] = 2
		if initial_status==2 or not check_sat(activation_pattern, properties, nnet):
			continue
		else:
			activation_pattern[layerNum][candidate] = initial_status
	return activation_pattern			



