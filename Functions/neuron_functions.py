import numpy as np
from numpy.linalg import norm


def electrode_neuron_inv_dist(num_electrodes, num_neurons, electrode_positions, neuron_positions, reach,
                              dimensionnality):
    """
    Returns a 2D-array of the electrodes-neurons inverse distances.
    The distances are null if greater than the reach.
    """
    distances           = np.zeros((num_electrodes, num_neurons))
    electrode_positions = np.resize(electrode_positions, (num_electrodes, dimensionnality))
    for e_index in range(num_electrodes):
        e_array = np.resize(electrode_positions[e_index], (num_neurons, dimensionnality))
        distances[e_index, :] = norm(neuron_positions-e_array, axis=1) #1D array of the distances between 
                                                                       #the electrode and the neurons
    inv_dist = np.power(distances, -1) #inverse distances
    return inv_dist


def electrode_grid(num_electrodes, network_dimensions, reach):
    """
    Return a grid of electrodes which reaches are within the network.
    """
    if num_electrodes % 2 == 0:
        raise ValueError("Must have an odd number of electrodes")
    
    linear_num = int(np.sqrt(num_electrodes))

    width  = (network_dimensions[1]-2*reach)/2
    height = (network_dimensions[0]-2*reach)/2

    horizontal_positions = np.resize((width/(linear_num//2))*np.arange(-(linear_num//2), linear_num//2+1), (num_electrodes))
    vertical_positions   = []
    for i in range(linear_num):
        for j in range(linear_num):
            vertical_positions.append(i)
    vertical_positions = (height/(linear_num//2))*(np.array(vertical_positions)-linear_num//2)

    
    electrode_positions       = np.zeros((3, num_electrodes))
    electrode_positions[1, :] = horizontal_positions
    electrode_positions[0, :] = vertical_positions

    return electrode_positions


def spike_rate(spiketrain, beginning, ending):
    '''Returns the spike rate of a neuron of spike train spiketrain
    between the time points beginning and ending (in ms).'''
    c = 0.
    for t in spiketrain:
        if t >= beginning and t < ending:
            c+=1.
    return 1000.*c/(ending-beginning)