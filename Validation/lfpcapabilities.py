import sciunit


class ProducesLocalFieldPotential(sciunit.Capability):
    def produce_local_field_potential():
        """
        The implementation of this method should return a Local Field Potential
        in shape of an N-D array, with N=1 if there is one electrode
        and N=2 if there are many electrodes.
        """
        raise NotImplementedError("Must implement produce_local_field_potential.")


class ProducesConductance(sciunit.Capability):
    """
    Indicates that the model produces the synaptic conductances.
    """
    
    def get_conductance():
        """
        The implementation of this method should return the conductances in shape of a
        neo.core.analogsignal.AnalogSignal.
        """
        raise NotImplementedError("Must implement get_conductance.")


class ProducesPositions(sciunit.Capability):
    """
    Indicates that the model produces the positions of the neurons.
    """

    def get_positions():
        """
        The implementation of this method should return the positions of the neurons in shape of a 2-D array,
        the first axis corresponding to the index, the other to the x, y, z coordinates (if z exists).
        """
        return NotImplementedError("Must implement get_positions.")
    
    def assign_positions():
        """
        The implementation of this method should return the positions of the neurons in shape of a 2-D array,
        just like the get_positions method, exept that it computes the positions itself
        """
        return NotImplementedError("Must implement assign_positions.")