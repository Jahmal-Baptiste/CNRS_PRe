import numpy as np
from numpy.linalg import norm


def electrode_neuron_inv_dist(num_electrodes, num_neurons, electrode_positions, neuron_positions, reach,
                              dimensionnality):
    """
    Returns a 2D-array of the electrodes-neurons inverse distances.
    The distances are null if greater than the reach.
    """
    distances = np.zeros((num_electrodes, num_neurons))
    for e_index in range(num_electrodes):
        ones_array = np.ones((num_neurons, dimensionnality))                 #as many lines as neurons
        e_array    = np.multiply(ones_array, electrode_positions[e_index])   #matrix of the electrode position repeated
        distances[e_index, :] = norm(neuron_positions-e_array, axis=1)       #1D array of the distances between 
                                                                             #the electrode and the neurons

    inv_dist   = np.power(distances, -1)                                     #inverse distances
    #valid_dist = np.heaviside(reach*np.ones(distances.shape) - distances, 1) #normalized distances within the reach
    #inv_dist   = np.multiply(inv_dist, valid_dist)                           #selected distances (within the reach)
    
    return inv_dist


def spike_rate(spiketrain, beginning, ending):
    '''Returns the spike rate of a neuron of spike train spiketrain
    between the time points beginning and ending (in ms).'''
    c = 0.
    for t in spiketrain:
        if t >= beginning and t < ending:
            c+=1.
    return 1000.*c/(ending-beginning)