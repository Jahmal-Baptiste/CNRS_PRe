import sciunit
import neuronunit
from neuronunit.capabilities import ProducesMembranePotential, ProducesSpikes
from sciunit.scores import ZScore

import pickle
import neo
import quantities as pq
import numpy as np

import sys
sys.path.append("..") #not the best way to modify sys.path...
from Validation.capabilities import ProducesLocalFieldPotential
import Fonctions.math_functions as mf


class Vm_LFP_CorrelationTest(sciunit.Test):
    """
    Tests if the computed LFP has the right correlation level with the Vm in two states:
    - low-activity state;
    - asynchronous state.
    
    The test first guesses in which state the network is then conducts the proper score accordingly.
    Indeed the correlation is different according to the state in which the network is:
    - strongly negative at low lag values for the low-activity state;
    - practically null for the asynchronous state.
    
    The test is adapted for monkeys' primary visual cortex performing a visual fixation task.
    Source:
    Tan, A., Chen, Y., Scholl, B., Seidemann, E. & Priebe, J.N. (2014, Mai 8). Sensory stimulation shifts visual cortex
    from synchronous to asynchronous states. Nature, 509, 226-229. doi:10.1038/nature13159
    """

    def __init__(self, network_state=""):
        self.network_state = network_state #state of the neuronal activity (either "low-activity" or "asynchronous")
        return super(Vm_LFP_CorrelationTest, self).__init__(network_state)
    
    required_capabilities = (ProducesMembranePotential, ProducesLocalFieldPotential,)

    score_type = ZScore

    def validate_observation(self, observation):
        """
        Checks if the network observation is in the expected state (low-activity for a blank fixation experiment or
        asynchronous for a sinusoidal stimulus).
        If the network is not in the expected state the user is asked if he still wants to continue with the test.
        """
        if self.network_state == "low-activity":
            raise NotImplementedError("Must implement the validate_observation method for the low-activity state.")
        elif self.network_state == "asynchonous":
            raise NotImplementedError("Must implement the validate_observation method for the asynchronous state.")
        else:
            raise ValueError("The network state is not defined.")
    
    def generate_prediction(self, model):
        """It is rather straight foward."""
        return model.produce_vm_LFP_correlation(start=600, duration=1000, dt=0.1)

    def compute_score(self, observation, prediction):
        if self.network_state == "low-activity":
            raise NotImplementedError("Must implement the compute-score method for the low-activity state.")
        elif self.network_state == "asynchonous":
            raise NotImplementedError("Must implement the compute_score method for the asynchronous state.")
        else:
            raise ValueError("The network state is not identified.")