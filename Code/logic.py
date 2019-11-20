import numpy as np


# asssumes that activation_pattern implies properties
def findMinimal (activation_pattern, properties):
	assert(isSatisfied(activation_pattern, properties))

	net_dim = activation_pattern.shape

	m_activation_pattern = np.copy(activation_pattern)

	for layer in range(net_dim[0]-1, -1, -1):
		
		num_neurons = (m_activation_pattern[layer].shape)[0]
		m_activation_pattern[layer] = np.repeat(2, num_neurons)
		
		if isSatisfied(m_activation_pattern, properties):
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
	# m_activation_pattern = np.copy(activation_pattern)
	while len(candidates)!=0:

		candidate = getNext(candidates, activation_pattern, properties, layerNum)
		initial_status = m_activation_pattern[layer][candidate]
		
		m_activation_pattern[layer][candidate] = 2
		
		if initial_status==2 or isSatisfied(m_activation_pattern, properties):
			np.delete(candidates, candidate)
			continue
		else:
			m_activation_pattern[layer][candidate] = initial_status
			return m_activation_pattern



