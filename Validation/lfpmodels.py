import sys, os

import sciunit
import neuronunit
from neuronunit.capabilities import ProducesMembranePotential, ProducesSpikes

import pickle
import neo
import quantities as pq
import numpy as np
from scipy.signal import coherence, hanning
from scipy.fftpack import fft
from itertools import chain
import os.path


from Validation.lfpcapabilities import ProducesLocalFieldPotential, ProducesConductance
from Functions import math_functions as mf
from Functions import neuron_functions as nf
from Functions import crosscorrelation as crsscorr
from Functions import filters as filt


class CoulombModel(sciunit.Model, ProducesLocalFieldPotential, ProducesMembranePotential,
                   ProducesConductance, ProducesSpikes):
    """
    A model of LFP computation that relies on the Coulomb law.
    It also checks if positional data is available. If not it assigns positions to the neurons (randomly).
    """

    def __init__(self, name=None, network_model="VA", space_dependency=False, dimensionnality=3,
                 dimensions=np.array([0.006, 0.006, 0.]), reach=0.001,
                 electrode_positions=np.array([[0.], [0.], [0.]]), sigma=0.3):
        self.name                = name
        self.network_model       = network_model       #Voggels-Abbott for the moment
        self.space_dependency    = space_dependency    #Boolean indicating if the neurons' positions are available
        self.dimensionnality     = dimensionnality     #dimensionnality of the network - either 2 or 3 D
        self.dimensions          = dimensions          #3D-array: leght, width and height of the network (in m)
        self.reach               = reach               #reach of the LFP (in m)
        elec_pos = np.transpose(electrode_positions)   #to have the the coordinates along the 0 axis, as opposed to the input state
        self.electrode_positions = elec_pos            #positions of the electrodes (in m)
        self.dt                  = 0.1                 #time step
        self.sigma               = sigma*1e6           #parameter in the Coulomb law's formula (in uS/m)
        self.directory_ABSPATH   = os.path.abspath("")
        
        self.num_neurons         = 0                   #number of neurons computed - will be properly initialized during LFP computation
        self.exc_counted         = False               #bool that tells if the excitatory neurons have been counted
        self.inh_counted         = False               #bool that tells if the inhibitatory neurons have been counted

        self.seed                = 42                  #seed for "random selections" 
        self.ratio               = False               #bool that tells if the ratio 4:1 is to be respected when selecting the neurons

        ### COMPUTATION OF THE NUMBER OF SEGMENTS
        self.set_directory_path()
        if self.network_model == "T2":
            self.num_trials = 1                        #number of trials for a same experiment
        elif self.network_model == "VA":
            self.num_trials = 1
        else:
            raise ValueError("Only the T2 and the Voggels-Abott models are supported.")
        
        ### VERIFICATION IF THE ELECTRODE(S) IS/ARE "INSIDE" THE NETWORK
        for e_pos in electrode_positions:
            if max(abs(e_pos))+self.reach > self.dimensions[0]/2.:
                raise ValueError("Wrong electrode position! Must have its reach zone in the network.")
        
        return super(CoulombModel, self).__init__(name) #MUST FINISH THIS LINE (name=name, etc.)
    
    #================================================================================================================
    #== methods related to the file selection =======================================================================
    #================================================================================================================

    def set_directory_path(self, date="20190718"):
        if self.network_model == "VA":
            parent_directory="./Examples/Results/"
            
            directory_path = parent_directory + date
            
            if not os.path.exists(directory_path):
                sys.exit("Directory does not exist!")
            self.directory_ABSPATH = os.path.abspath(directory_path)

        elif self.network_model == "T2":
            directory_path = "./T2/ThalamoCorticalModel_data_size_____/"

            if not os.path.exists(directory_path):
                sys.exit("Directory does not exist!")
            self.directory_ABSPATH = os.path.abspath(directory_path)
        else:
            raise NotImplementedError("Only the T2 and the Voggels-Abott models are supported.")
            
    
    def get_file_path(self, segment_number="0", time="201157", neuron_type=""):
        if self.network_model == "VA":
            if neuron_type == "":
                raise ValueError("Must specify a neuron type.")
            
            date      = os.path.basename(self.directory_ABSPATH)
            file_path = str(self.directory_ABSPATH) + "/VAbenchmarks_COBA_{0}_neuron_np1_{1}-{2}.pkl".format(neuron_type,
                                                                                                             date, time)
            if not os.path.exists(file_path):
                sys.exit("File name does not exist! (Try checking the time argument.)")
        
        elif self.network_model == "T2":
            file_path = str(self.directory_ABSPATH) + "/Segment{}.pickle".format(segment_number)

            if not os.path.exists(file_path):
                sys.exit("File name does not exist!")
        else:
            raise NotImplementedError("Only the T2 and the Voggels-Abott models are supported.")
        return file_path
    

    
    #================================================================================================================
    #== methods related to raw available data =======================================================================
    #================================================================================================================
    
    
    def get_membrane_potential(self, trial=0, experiment="sin_stim"):
        """
        Returns a neo.core.analogsignal.AnalogSignal representing the membrane potential of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        ### VA MODEL ###
        if self.network_model == "VA":
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_exc = analogsignal
            if self.exc_counted == False:
                self.num_neurons += vm_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_inh = analogsignal
            if self.inh_counted == False:
                self.num_neurons += vm_inh.shape[1]
                self.inh_counted  = True

            ### ALL NEURONS
            vm_array = np.concatenate((np.array(vm_exc), np.array(vm_inh)), axis=1)
            vm       = neo.core.AnalogSignal(vm_array, units=vm_exc.units, t_start=vm_exc.t_start,
                                             sampling_rate=vm_exc.sampling_rate)
        ### T2 MODEL ###
        else:
            if experiment == "sin_stim":
                data_int = 0
            elif experiment == "blank_stim":
                data_int = 5
            else:
                raise ValueError("The experiment argument must be either 'sin_stim' or 'blank_stim'.")
            
            ### EXCITATORY NEURONS
            seg_num     = str(20*(trial+1)+data_int+2)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file, encoding="latin1")
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_exc = analogsignal
            if self.exc_counted == False:
                self.num_neurons += vm_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            seg_num     = str(20*(trial+1)+data_int+1)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file, encoding="latin1")
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_inh = analogsignal
            # selection of the inhibitory neurons to have 1/4 of the amount of excitatory neurons
            if self.ratio:
                np.random.seed(self.seed)
                inh_list = mf.random_list(vm_exc.shape[1]//4, vm_inh.shape[1]-1, minimum=0)
                vm_inh   = vm_inh[:, inh_list]
            
            if self.inh_counted == False:
                self.num_neurons += vm_inh.shape[1]
                self.inh_counted  = True
            
            ### ALL NEURONS
            vm_array = np.concatenate((np.array(vm_exc), np.array(vm_inh)), axis=1)
            vm       = neo.core.AnalogSignal(vm_array, units=vm_exc.units, t_start=vm_exc.t_start,
                                             sampling_rate=vm_exc.sampling_rate)
        
            # replacement of nan values by zeros
            vm_nan     = np.isnan(vm)
            vm[vm_nan] = 0*vm.units
            
            # replacement of nan/zero values by the runnig average value of the neurons that have non-nan values till the end
            vm_valid_indexes = ~np.isnan(vm[-1, :])
            average_vm       = np.reshape(np.average(vm, axis=1, weights=vm_valid_indexes), (vm.shape[0], 1))
            average_vm       = np.multiply(average_vm, np.ones(vm.shape))
            vm[vm_nan]       = average_vm[vm_nan]

            # print of the number of "valid" neurons
            #unique, counts   = np.unique(vm_valid_indexes, return_counts=True)
            #print(dict(zip(unique, counts)))
        return vm



#====================================================================================================================

    def get_conductance(self, trial=0, experiment="sin_stim", max_time=300):
        """
        Returns a neo.core.analogsignal.AnalogSignal representing the synaptic conductance of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if self.network_model == "VA":
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_exc':
                    gsyn_exc = analogsignal
            if self.exc_counted == False:
                self.num_neurons += gsyn_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_inh':
                    gsyn_inh = analogsignal
            if self.inh_counted == False:
                self.num_neurons += gsyn_inh.shape[1]
                self.inh_counted  = True

            ### ALL NEURONS
            gsyn_array = np.concatenate((np.array(gsyn_exc), np.array(gsyn_inh)), axis=1)
            gsyn       = neo.core.AnalogSignal(gsyn_array, units=gsyn_exc.units, t_start=gsyn_exc.t_start,
                                               sampling_rate=gsyn_exc.sampling_rate)
        else:
            if experiment == "sin_stim":
                data_int = 0
            elif experiment == "blank_stim":
                data_int = 5
            else:
                raise ValueError("The experiment argument must be either 'sin_stim' or 'blank_stim'.")
            
            ### EXCITATORY NEURONS
            seg_num     = str(20*(trial+1)+data_int+2)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file, encoding="latin1")
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_exc':
                    gsyn_exc_exc = analogsignal
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_inh':
                    gsyn_exc_inh = analogsignal
            
            gsyn_exc            = np.array(np.resize(gsyn_exc_exc, (gsyn_exc_exc.shape[0], gsyn_exc_exc.shape[1], 2)))
            gsyn_exc_inh_array  = np.array(gsyn_exc_inh)
            gsyn_exc[:, :, 1]   = gsyn_exc_inh_array
            conductance_weights = np.array([4., 1.])
            gsyn_exc            = np.average(gsyn_exc, axis=2, weights=conductance_weights)

            if self.exc_counted == False:
                self.num_neurons += gsyn_exc.shape[1]
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            seg_num     = str(20*(trial+1)+data_int+1)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file, encoding="latin1")
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_exc':
                    gsyn_inh_exc = analogsignal
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_inh':
                    gsyn_inh_inh = analogsignal
            
            gsyn_inh            = np.array(np.resize(gsyn_inh_exc, (gsyn_inh_exc.shape[0], gsyn_inh_exc.shape[1], 2)))
            gsyn_inh_inh_array  = np.array(gsyn_inh_inh)
            gsyn_inh[:, :, 1]   = gsyn_inh_inh_array
            conductance_weights = np.array([4., 1.])
            gsyn_inh            = np.average(gsyn_inh, axis=2, weights=conductance_weights)

            # selection of the inhibitory neurons to have 1/4 of the amount of excitatory neurons
            if self.ratio:
                np.random.seed(self.seed)
                inh_list = mf.random_list(gsyn_exc.shape[1]//4, gsyn_inh.shape[1]-1, minimum=0)
                gsyn_inh = gsyn_inh[:, inh_list]

            if self.inh_counted == False:
                self.num_neurons += gsyn_inh.shape[1]
                self.inh_counted  = True
            
            ### ALL NEURONS
            gsyn_array = np.concatenate((gsyn_exc, gsyn_inh), axis=1)
            gsyn       = neo.core.AnalogSignal(gsyn_array, units=gsyn_exc_exc.units, t_start=gsyn_exc_exc.t_start,
                                             sampling_rate=gsyn_exc_exc.sampling_rate)
            
            # replacement of the nan values by zeros
            gsyn_nan       = np.isnan(gsyn)
            gsyn[gsyn_nan] = 0*gsyn.units

            # replacement of the nan/zero values by the running average value of the neurons that remain non-nan till the end
            gsyn_valid_indexes = ~np.isnan(gsyn[-1, :])
            average_gsyn       = np.reshape(np.average(gsyn, axis=1, weights=gsyn_valid_indexes), (gsyn.shape[0], 1))
            average_gsyn       = np.multiply(average_gsyn, np.ones(gsyn.shape))
            gsyn[gsyn_nan]     = average_gsyn[gsyn_nan]

            # print of the number of "valid" neurons
            #unique, counts     = np.unique(gsyn_valid_indexes, return_counts=True)
            #print(dict(zip(unique, counts)))

        return gsyn
    


#====================================================================================================================
    
    def get_spike_trains(self, trial=0, experiment="sin_stim"):
        """
        Returns a list of neo.core.SpikeTrain elements representing the spike trains of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if self.network_model == "VA":
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            spiketrains_exc = seg.spiketrains
            if self.exc_counted == False:
                self.num_neurons += len(spiketrains_exc)
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            block       = pickle.load(PyNN_file)
            seg         = block.segments[trial] #chosen segment
            spiketrains_inh = seg.spiketrains
            if self.inh_counted == False:
                self.num_neurons += len(spiketrains_inh)
                self.inh_counted  = True

            ### ALL NEURONS
            spiketrains = spiketrains_exc + spiketrains_inh
        else:
            if experiment == "sin_stim":
                data_int = 0
            elif experiment == "blank_stim":
                data_int = 5
            else:
                raise ValueError("The experiment argument must be either 'sin_stim' or 'blank_stim'.")

            ### EXCITATORY NEURONS
            seg_num     = str(20*(trial+1)+data_int+2)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file, encoding="latin1")
            spiketrains_exc = seg.spiketrains
            if self.exc_counted == False:
                self.num_neurons += len(spiketrains_exc)
                self.exc_counted  = True
            
            ### INHIBITORY NEURONS
            seg_num     = str(20*(trial+1)+data_int+1)
            file_path   = self.get_file_path(segment_number=seg_num)
            PyNN_file   = open(file_path, "rb")
            seg         = pickle.load(PyNN_file, encoding="latin1")
            spiketrains_inh = seg.spiketrains

            # selection of the inhibitory neurons to have 1/4 of the amount of excitatory neurons
            if self.ratio:
                np.random.seed(self.seed)
                inh_list        = mf.random_list(len(spiketrains_exc)//4, len(spiketrains_inh)-1, minimum=0)
                spiketrains_inh_sel = []
                for i in inh_list:
                    spiketrains_inh_sel.append(spiketrains_inh[i])
            else:
                spiketrains_inh_sel = spiketrains_inh

            if self.inh_counted == False:
                self.num_neurons += len(spiketrains_inh_sel)
                self.inh_counted  = True
            
            ### ALL NEURONS
            spiketrains = spiketrains_exc + spiketrains_inh_sel
        return spiketrains



#====================================================================================================================

    def get_spike_train(self, trial=0):
        global_spiketrain = list(chain.from_iterable(spiketrain for spiketrain in self.get_spike_trains(trial=trial)))
        global_spiketrain.sort()
        return global_spiketrain
    


#====================================================================================================================

    def get_positions(self):
        """
        Returns the 2D-array giving the neurons' positions.
        """
        if self.space_dependency == False:
            positions = self.assign_positions()

            if self.exc_counted == False and self.inh_counted == False:
                self.num_neurons = positions.shape[1]
                self.exc_counted = True
                self.inh_counted = True
        else:
            if self.network_model == "T2":
                ### EXCITATORY NEURONS
                positions_path_exc = str(self.directory_ABSPATH) + "/id_positions_V1_Exc_L4.pickle"
                positions_file_exc = open(positions_path_exc, "rb")
                positions_dict_exc = pickle.load(positions_file_exc, encoding="latin1")
                exc_keys          = list(positions_dict_exc.keys())
                num_exc           = len(exc_keys)
                positions_exc     = np.zeros((num_exc, 3))
                for k in range(num_exc):
                    positions_exc[k, :] = np.array(positions_dict_exc[exc_keys[k]])
                
                ### INHIBITORY NEURONS
                positions_path_inh = str(self.directory_ABSPATH) + "/id_positions_V1_Inh_L4.pickle"
                positions_file_inh = open(positions_path_inh, "rb")
                positions_dict_inh = pickle.load(positions_file_inh, encoding="latin1")
                inh_keys          = list(positions_dict_inh.keys())
                if self.ratio:
                    # selection of the inhibitory neurons to have 1/4 of the amount of excitatory neurons
                    num_inh   = num_exc//4
                    np.random.seed(self.seed)
                    inh_list = mf.random_list(num_inh, len(inh_keys)-1, minimum=0)
                    positions_inh = np.zeros((num_inh, 3))
                    for k in range(num_inh):
                        positions_inh[k, :] = np.array(positions_dict_inh[inh_keys[inh_list[k]]])
                else:
                    num_inh   = len(inh_keys)
                    positions_inh = np.zeros((num_inh, 3))
                    for k in range(num_inh):
                        positions_inh[k, :] = np.array(positions_dict_inh[inh_keys[k]])
                
                ### ALL NEURONS
                positions = np.concatenate((positions_exc, positions_inh), axis=0)
                positions = 1e-3*positions #conversion into meters

                if self.exc_counted == False and self.inh_counted == False:
                    self.num_neurons = num_exc + num_inh
                    self.exc_counted = True
                    self.inh_counted = True
            
            else:
                raise ValueError("Only the T2 model has available positions.")
        return positions



#====================================================================================================================

    def assign_positions(self):
        """
        Function that assigns positions to the neurons if they do not have.
        Only works if they have a 2D structure.
        """
        if self.exc_counted == False and self.inh_counted == False:
            self.num_neurons = len(self.get_spike_trains(trial=0))
            self.exc_counted = True
            self.inh_counted = True
        positions = np.multiply(self.dimensions, np.random.rand(self.num_neurons, self.dimensionnality)-0.5)
        return positions



    #================================================================================================================
    #== LFP related methods =========================================================================================
    #================================================================================================================
    
    def produce_local_field_potential(self, trial=0):
        """
        Calculates and returns the 2D-array of the LFP.
        The first dimension corresponds to the electrodes.
        """
        vm                = self.get_membrane_potential(trial=trial)
        gsyn              = self.get_conductance(trial=trial)
        neuron_positions  = self.get_positions()
        
        num_time_points   = vm.shape[0]
        num_electrodes    = self.electrode_positions.shape[0]
        
        current_array     = np.multiply(vm, gsyn)
        big_current_array = np.resize(current_array, (num_time_points, self.num_neurons, num_electrodes))
        big_current_array = np.transpose(big_current_array, (2, 0, 1))

        inv_dist     = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                    self.electrode_positions, neuron_positions,
                                                    self.reach, self.dimensionnality)
        valid_dist   = np.heaviside(inv_dist - 1/self.reach, 1) #normalized distances within the reach
        inv_dist     = np.multiply(inv_dist, valid_dist)        #selected distances (within the reach)
        big_inv_dist = np.resize(inv_dist, (num_electrodes, self.num_neurons, num_time_points))
        big_inv_dist = np.transpose(big_inv_dist, (0, 2, 1))

        indiv_LFP   = np.multiply(big_current_array, big_inv_dist)
        LFP         = np.sum(indiv_LFP, axis=2)/(4*np.pi*self.sigma)
        time_points = vm.times

        return LFP, time_points
    


    #================================================================================================================
    #== test related methods ========================================================================================
    #================================================================================================================
    
    def produce_vm_LFP_correlation(self, trial=0, start=0, duration=1000):
        """
        Calculates the correlation between the Vm of the closest neuron to the (first) electrode and the LFP signal
        recorded at this electrode.
        Returns the correlation and the corresponding lag (in ms).
        The relevant data is supposed to be 1s long.
        """
        print("\nCORRELATION FUNCTION\n")
        start_index    = int(start/self.dt)
        duration_index = int(duration/self.dt)

        vm               = self.get_membrane_potential(trial=trial)
        neuron_positions = self.get_positions()

        num_electrodes = self.electrode_positions.shape[0]
        inv_dist       = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                      self.electrode_positions, neuron_positions,
                                                      self.reach, self.dimensionnality)[0, :]
        closest_neuron = np.argmax(inv_dist)
        selected_vm    = np.reshape(vm[start_index:start_index+duration_index+1, closest_neuron], (duration_index+1,))
        selected_LFP   = np.reshape(self.produce_local_field_potential(trial=trial)[0][0,
                                                                                       start_index:start_index+duration_index+1],
                                    (duration_index+1,))
        
        corr             = crsscorr.constwindowcorrelation(selected_vm, selected_LFP)
        corr_time_points = np.arange(-duration/2, duration/2+self.dt, self.dt)
        return corr, corr_time_points



#====================================================================================================================

    def produce_vm_LFP_zerolagcorrelations(self, start=0, duration=1000,
                                           trial_average=True, trial=0, withinreach=True):
        """
        Calculates the zero-lag correlations between the neurons' membrane potentials and the LFP.
        Interesting plots to do with this data can be:
        - histogram of the correlation distribution;
        - confrontation of the correlation values between a non-stimulated and stimulated state (for the same neurons).
        The trial_average boolean tells if the correlations have to be averaged over the trials.
        If not, the chosen trial is trial.
        """
        print("\nZEROLAG CORRELATION FUNCTION\n")
        start_index    = int(start/self.dt)
        duration_index = int(duration/self.dt)

        if trial_average:
            trials     = self.num_trials
            trial_list = [k for k in range(self.num_trials)]
        else:
            trials     = 1
            trial_list = [trial]

        for iteration_trial in trial_list:
            print("  Trial " + str(iteration_trial) + "...")

            print("  Computation of the LFP...")
            vm  = self.get_membrane_potential(trial=iteration_trial)
            vm  = vm[start_index:start_index+duration_index+1, :]
            LFP = np.reshape(self.produce_local_field_potential(trial=iteration_trial)[0][0, start_index:start_index+duration_index+1],
                             (duration_index+1,))

            def zerolagcorrelationtoLFP(v):
                return np.corrcoef(v, LFP)[0, 1]
            
            ### ELIMINATION OF THE CONTRIBUTION OF NEURONS THAT ARE OUT OF THE REACH ZONE
            if withinreach:
                print("    Elimination of the contribution of the neurons out of the electrode's reach...")
                num_electrodes     = self.electrode_positions.shape[0]
                neuron_positions   = self.get_positions()
                inv_dist           = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                                  self.electrode_positions, neuron_positions,
                                                                  self.reach, self.dimensionnality)[0, :]
                valid_neurons         = np.heaviside(inv_dist - 1/self.reach, 1)
                valid_neurons_indexes = np.argwhere(valid_neurons > 0)
                valid_neurons_num     = valid_neurons_indexes.shape[0]
                vm_valid              = np.zeros((duration_index+1, valid_neurons_num))*vm.units
                for k in range(valid_neurons_num):
                    vm_valid_col      = np.reshape(vm[:, valid_neurons_indexes[k, 0]], duration_index+1)
                    vm_valid[:, k]    = vm_valid_col
                vm                    = vm_valid
                print("    Neurons left: " + str(vm.shape[1]))
            
            print("    Computation of the zerolag correlations...")
            zerolagcorrelations_array = np.zeros((trials, vm.shape[1]))
            zerolagcorrelations_array[iteration_trial, :] = np.apply_along_axis(zerolagcorrelationtoLFP, arr=vm, axis=0)

        zerolagcorrelations = np.average(zerolagcorrelations_array, axis=0)

        print("")
        return zerolagcorrelations #if withinreach==True, neurons that are out of the reach zone have a null correlation with the LFP



#====================================================================================================================

    def produce_vm_LFP_meancoherence(self, trial=0, withinreach=True, start=0, duration=1000):
        """
        Calculates the mean coherence between the neurons' membrane potentials and the LFP.
        returns the mean coherence, the corresponding frequencies (in Hz) and the standard deviation error for each
        coherence value.
        The relevant data is supposed to be 1s long.
        """
        print("\nMEAN COHERENCE FUNCTION\n")
        start_index    = int(start/self.dt)
        duration_index = int(duration/self.dt)

        print("    Computation of the LFP...")
        vm  = self.get_membrane_potential(trial=trial)
        vm  = vm[start_index:start_index+duration_index+1, :]
        LFP = np.reshape(self.produce_local_field_potential(trial=trial)[0][0, start_index:start_index+duration_index+1],
                         (duration_index+1, 1))
        
        ### ELIMINATION OF THE CONTRIBUTION OF NEURONS THAT ARE OUT OF THE REACH ZONE
        if withinreach:
            print("    Elimination of the contribution of the neurons out of the electrode's reach...")
            num_electrodes   = self.electrode_positions.shape[0]
            neuron_positions = self.get_positions()
            inv_dist         = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                            self.electrode_positions, neuron_positions,
                                                            self.reach, self.dimensionnality)[0, :]
            valid_neurons         = np.heaviside(inv_dist - 1/self.reach, 1)
            valid_neurons_indexes = np.argwhere(valid_neurons > 0)
            valid_neurons_num     = valid_neurons_indexes.shape[0]
            vm_valid              = np.zeros((duration_index+1, valid_neurons_num))*vm.units
            for k in range(valid_neurons_num):
                vm_valid_col      = np.reshape(vm[:, valid_neurons_indexes[k, 0]], duration_index+1)
                vm_valid[:, k] = vm_valid_col
            vm                 = vm_valid
            print("    Neurons left: " + str(vm.shape[1]))

        print("    Computation of the coherence...")
        f, coherence_array = coherence(LFP, vm, axis=0, nperseg=int(2**12), fs=1000./self.dt)
        meancoherence      = np.average(coherence_array, axis=1)
        coherencestd       = np.std(coherence_array, axis=1)

        relevant_index = np.argwhere(f > 50.)[0, 0]
        f              = f[:relevant_index+1]
        meancoherence  = meancoherence[:relevant_index+1]
        coherencestd   = coherencestd[:relevant_index+1]
        print("")
        return meancoherence, f, coherencestd



#====================================================================================================================

    def produce_phase_lock_value(self, start=0, offset=250, duration=1000, N_max=100,
                                 trial_average=True, trial=0, withinreach=True):
        """
        Calculates the Phase-Lock value for the spikes occuring in a (duration-offset) ms period of time.
        The neurons are supposed to be excitated by a sinusoidal input of 1s, starting offset ms before the
        selected epoch.
        Returns the Phase-Lock value and the corresponding frequencies.
        The trial_average boolean tells if the Phase-Lock value has to be averaged over the trials.
        If not, the chosen trial is trial.
        """
        print("\nPHASE-LOCK VALUE FUNCTION\n")
        if trial_average:
            trials     = self.num_trials
            trial_list = [k for k in range(self.num_trials)]
        else:
            trials     = 1
            trial_list = [trial]
        
        fs           = 1000./self.dt                      #sampling frequency
        window       = 150                                #ms, size of the window in which the LFP will have its Fourier transformations
        window_index = int(window/self.dt)
        window_width = window_index//2
        w            = hanning(window_index+1)            #150 ms window with a 0,1 ms interval
        PLv_array    = np.zeros((trials, window_index+1),
                                dtype=float)              #multi-trial Phase-Lock value array, empty for the moment

        for iteration_trial in trial_list:
            print("  Trial " + str(iteration_trial) + "...")

            print("    Computation of the LFP...")
            LFP             = self.produce_local_field_potential(trial=iteration_trial)[0][0, :]
            
            print("    Loading of the spikes...")
            if withinreach:
                spiketrains = self.get_spike_trains(trial=iteration_trial)

                print("      Elimination of the contribution of the neurons out of the electrode's reach...")
                num_electrodes     = self.electrode_positions.shape[0]
                neuron_positions   = self.get_positions()
                inv_dist           = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                                  self.electrode_positions, neuron_positions,
                                                                  self.reach, self.dimensionnality)[0, :]
                valid_neurons         = np.heaviside(inv_dist - 1/self.reach, 1)
                valid_neurons_indexes = np.argwhere(valid_neurons > 0)
                valid_neurons_num     = valid_neurons_indexes.shape[0]
                spiketrains_sel       = []
                for k in range(valid_neurons_num):
                    spiketrains_sel.append(spiketrains[valid_neurons_indexes[k, 0]])
                print("      Neurons left: " + str(len(spiketrains_sel)))
                spiketrain = np.array(list(chain.from_iterable(spiketrain for spiketrain in spiketrains_sel)))
            else:
                spiketrain = np.array(self.get_spike_train(trial=iteration_trial))

            print("    Selection of the spikes in the proper time window...")
            selected_spikes = np.multiply(spiketrain, mf.door(spiketrain, start+offset, start+duration-window//2))
            selected_spikes = selected_spikes[selected_spikes > 0]
            N_s             = min(selected_spikes.shape[0], N_max)  #security measure
            rnd_spike_list  = mf.random_list(N_s, selected_spikes.shape[0]-1, minimum=0)
            if N_s < N_max:
                sys.exit("Not enough spikes in this segment (or at least in the time window).")

            print("    Computation of the Phase-Lock value...")
            for t_index in rnd_spike_list:
                t_s       = selected_spikes[t_index]                                         #time of spike occurence
                t_s_index = int(10*t_s)                                                      #corresponding index for the arrays
                LFP_s     = LFP[t_s_index - window_width : t_s_index + window_width+1]       #Non-filtered version
                wLFP_s    = np.multiply(w, LFP_s)                                            #centered LFP multiplied by the Hanning window
                FT_s      = fft(wLFP_s)                                                      #Fourier transform of this weighted LFP
                nFT_s     = np.divide(FT_s, np.abs(FT_s))                                    #normalized Fourier transform
                nFT_s     = np.real(nFT_s)
                PLv_array[iteration_trial, :] = np.add(PLv_array[iteration_trial, :], nFT_s) #contribution to the PLv added

            PLv_array[iteration_trial, :]  = np.abs(PLv_array[iteration_trial, :])/N_s       #normalized module, according to the paper
            
        PLv  = np.average(PLv_array, axis=0)                          #trial-average
        fPLv = (fs/PLv.shape[0])*np.arange(PLv.shape[0], dtype=float) #frequencies of the PLv

        min_index = np.argwhere(fPLv < 20.)[-1, 0]
        max_index = np.argwhere(fPLv > 100.)[0, 0]
        fPLv      = fPLv[min_index:max_index+1]
        PLv       = PLv[min_index:max_index+1]
        print("")
        return PLv, fPLv



#====================================================================================================================

    def produce_spike_triggered_LFP(self, start=0, window_width=400, duration=1000,
                                    discrim_dist_list=[4e-4, 8e-4, 1.2e-3, 1.6e-3],
                                    trial_average=True, trial=0):
        """
        Calculates the spike-triggered average of the LFP (stLFP) and arranges the results relative to the distance
        from the electrode. The distances discriminating the neurons are (in mm): 0.4, 0.8, 1.2, 1.4 and 1.6.
        Returns the stLFP for each distance interval and in a time interval around the spikes.
        The stLFP is a 2D-array with the first dimension corresponding to the distance and the second, the time.
        """
        print("\nSPIKE TRIGGERED LFP FUNCTION\n")
        discrim_dist     = np.array(discrim_dist_list)
        discrim_inv_dist = np.append(np.inf, np.power(discrim_dist, -1))
        discrim_inv_dist = np.append(discrim_inv_dist, 0.)

        num_electrodes   = self.electrode_positions.shape[0]
        neuron_positions = self.get_positions()
        inv_dist         = nf.electrode_neuron_inv_dist(num_electrodes, self.num_neurons,
                                                        self.electrode_positions, neuron_positions,
                                                        self.reach, self.dimensionnality)[0, :]
        
        discrim_indexes    = [[], [], [], [], []]
        num_dist_intervals = len(discrim_indexes)
        for i in range(num_dist_intervals):
            normalized_inv_dist_i = mf.door(inv_dist, discrim_inv_dist[i+1], discrim_inv_dist[i])
            indexes_i             = np.argwhere(normalized_inv_dist_i > 0)
            discrim_indexes[i]    = indexes_i.flatten().tolist()
        
        '''
        Now we have discriminated the neurons according to their distance from the electrode (information stored in
        their indexes in the list discrim_indexes), we can separate the stLFPs according to this criteria.
        '''

        if trial_average:
            trials     = self.num_trials
            trial_list = [k for k in range(self.num_trials)]
        else:
            trials     = 1
            trial_list = [trial]
        
        offset         = window_width/2
        window_index   = int(window_width/self.dt)
        window         = np.arange(-window_width/2, window_width/2+self.dt, self.dt)
        stLFP_array    = np.zeros((trials, num_dist_intervals, window_index+1))

        for iteration_trial in trial_list:
            ### LOOP ON THE TRIALS
            print("  Trial " + str(iteration_trial) + "...")

            print("    Loading of the spiketrains...")
            spiketrains = self.get_spike_trains(trial=iteration_trial)
            print("    Computation of the LFP...")
            LFP         = self.produce_local_field_potential(trial=iteration_trial)[0][0, :]

            for interval_index in range(num_dist_intervals):
                ### LOOP ON THE DISTANCE INTERVALS
                num_spikes          = 0
                for neuron_index in discrim_indexes[interval_index]:
                    ### LOOP ON THE NEURONS WITHIN A DISTANCE INTERVAL
                    spiketrain      = np.array(spiketrains[neuron_index])
                    selected_spikes = np.multiply(spiketrain, mf.door(spiketrain, start+offset, start+duration-offset))
                    selected_spikes = selected_spikes[selected_spikes > 0]
                    for t_s in selected_spikes:
                        ### LOOP ON THE SPIKES OF A GIVEN NEURON
                        t_s_index   = int(t_s/self.dt)
                        num_spikes += 1
                        stLFP_array[iteration_trial, interval_index, :] = np.add(
                                                            stLFP_array[iteration_trial, interval_index, :],
                                                            LFP[t_s_index-window_index//2:t_s_index+window_index//2+1])
                if num_spikes > 0:
                    stLFP_array[iteration_trial, interval_index, :] /= num_spikes
        
        stLFP = np.average(stLFP_array, axis=0) #trial-average computation
        print("")
        return stLFP, window