import sciunit
import neuronunit
from neuronunit.capabilities import ProducesMembranePotential, ProducesSpikes
from sciunit.scores import ZScore

import pickle
import neo
import quantities as pq
import numpy as np
from scipy.stats import skew

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

    def __init__(self, experiment=""):
        if experiment == "":
            raise ValueError('''The experiment is not specified (either "blank" or "stimulus").''')
        self.experiment = experiment #experiment conducted (either "blank" or "sinusoidal")
        return super(Vm_LFP_CorrelationTest, self).__init__(experiment)
    
    required_capabilities = (ProducesMembranePotential, ProducesLocalFieldPotential,)

    score_type = ZScore

    def validate_simulation(self, model, low_skew=0.3, high_skew=0.5):
        """
        Checks if the network observation is in the expected state (low-activity for a blank fixation experiment or \
        asynchronous for a sinusoidal stimulus).
        If the network is not in the expected state the user is asked if he still wants to continue with the test.
        Method based on the macaques' V1 characteristics (may not be useful for the cats' V1).
        """
        vm           = model.get_membrane_potential() #here I am using the simulated vm as the observation...
        #vm           = observation
        skew_array   = skew(vm)
        average_skew = np.average(skew_array)
        """
        There is a problem in the way I calculated the skew because I haven't gotten rid of the spike-related potentials, \
        which adds skew to the value that is meaningful...
        Anyway it seems (with the litterature I have read so far) that the cats are in a synchronous state before even \
        confronted to a visual stimulation so this first-hand test might be irrelevant.
        """

        if average_skew >= low_skew and average_skew <= high_skew:
            sys.exit("The observation skew does not allow a clear state identification.")

        if self.experiment == "blank" and average_skew < high_skew:
            answer = input("The skew of the data is inconsistent with a low-activity network state.\n\
                Do you want to continue the test? (y/n)\n")
            while answer not in ['y', 'n']:
                answer = input("Excuse me, I did not understand. (y/n)")
            if answer == 'n':
                sys.exit()
        elif self.experiment == "stimulus" and average_skew > low_skew:
            answer = input("The skew of the data is inconsistent with a synchronous network state.\n\
                Do you want to continue the test? (y/n)\n")
            while answer not in ['y', 'n']:
                answer = input("Excuse me, I did not understand. (y/n)")
            if answer == 'n':
                sys.exit()

    
    def generate_prediction(self, model):
        """It is rather straight foward."""
        return model.produce_vm_LFP_erolagcorrelations(start=600, duration=1000, dt=0.1,
                                                       trial_average=True, trial=0, withinreach=True)


    def compute_score(self, observation, prediction):
        if self.experiment == "blank":
            raise NotImplementedError("Must implement the compute-score method for the low-activity state.")
        elif self.network_state == "stimulus":
            raise NotImplementedError("Must implement the compute_score method for the asynchronous state.")
        else:
            raise ValueError("The network state is not identified.")