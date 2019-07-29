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
from pathlib import Path, PurePath

import sys
sys.path.append("..") #not the best way to modify sys.path but anyway...
from Validation.capabilities import ProducesLocalFieldPotential, ProducesConductance
import Fonctions.math_functions as mf
import Fonctions.neuron_functions as nf
import Fonctions.crosscorrelation as crsscorr
import Fonctions.filters as filt


class CoulombModel(sciunit.Model, ProducesLocalFieldPotential, ProducesMembranePotential,
                   ProducesConductance, ProducesSpikes):
    """
    A model of LFP computation that relies on the Coulomb law.
    It also checks if positional data is available. If not it assigns positions to the neurons (randomly or not).
    """

    def __init__(self, name=None, network_model="VA", space_dependency=False, dimensionnality=3,
                 dimensions=np.array([0.002, 0.002, 0.]), reach=0.001,
                 electrode_positions=np.array([[0.], [0.], [0.]]), sigma=0.3):
        self.name                = name
        self.network_model       = network_model       #Voggels-Abbott for the moment
        self.space_dependency    = space_dependency    #Boolean indicating if the neurons' positions are available
        self.dimensionnality     = dimensionnality     #dimensionnality of the network - either 2 or 3 D
        self.dimensions          = dimensions          #2,3D-array: leght, width and height (when exists) of the network (in m)
        self.reach               = reach               #reach of the LFP (in m)
        np.transpose(electrode_positions)              #to have the the coordinates along the 0 axis, as opposed to the input state
        self.electrode_positions = electrode_positions #positions of the electrodes (in m)
        self.sigma               = sigma               #parameter in the Coulomb law's formula (in S/m)
        self.directory_PUREPATH  = PurePath()
        
        for e_pos in electrode_positions:
            if max(abs(e_pos))+self.reach <= self.dimensions/2.:
                raise ValueError("Wrong electrode position! Must have its reach zone in the network.")
        
        return super(CoulombModel, self).__init__(name, network_model, space_dependency, dimensionnality,
                                                  dimensions, reach, electrode_positions, sigma) #MUST FINISH THIS LINE (name=name, etc.)
    
    #############################################
    ### methods related to raw available data ###
    #############################################

    def set_directory_path(self, parent_directory="../Exemples/Results/", date="20190718"):
        if self.network_model == "VA":
            directory_path = parent_directory + date
            directory_PATH = Path(directory_path)
            
            if not directory_PATH.exists():
                sys.exit("Directory does not exist!")
            self.directory_PUREPATH = PurePath(directory_path)
        else:
            raise NotImplementedError("Only the Voggels-Abbott model is implemented.")
            
    
    def get_file_path(self, time="201157", neuron_type=""):
        if self.network_model == "VA":
            if neuron_type == "":
                raise ValueError("Must specify a neuron type.")
            date      = self.directory_PUREPATH.parts[-1]
            file_path = str(self.directory_PUREPATH) + "VAbenchmarks_COBA_{0}_neuron_np1_{1}-{2}.pkl".format(neuron_type,
                                                                                                             date, time)
            file_PATH = Path(file_path)
            if not file_PATH.exists():
                sys.exit("File name does not exist! (Surely wrong time.)")
        else:
            raise NotImplementedError("Only the Voggels-Abbott model is implemented.")
        return file_path
    

    def get_membrane_potential(self, num_files=2):
        """
        Returns a neo.core.analogsignal.AnalogSignal representing the membrane potential of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if num_files == 2:
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_exc = analogsignal
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm_inh = analogsignal

            ### ALL NEURONS
            vm_array = np.concatenate(vm_exc, vm_inh, axis=1)
            vm       = neo.core.AnalogSignal(vm_array, units=vm_exc.units, t_start=vm_exc.t_start,
                                             sampling_rate=vm_exc.sampling_rate)
        else:
            neuron_type = "all"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'v':
                    vm = analogsignal
        return vm
    

    def get_conductance(self, num_files=2):
        """
        Returns a neo.core.analogsignal.AnalogSignal representing the synaptic conductance of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if num_files == 2:
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_exc':
                    gsyn_exc = analogsignal
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn_inh':
                    gsyn_inh = analogsignal

            ### ALL NEURONS
            gsyn_array = np.concatenate(gsyn_exc, gsyn_inh, axis=1)
            gsyn       = neo.core.AnalogSignal(gsyn_array, units=gsyn_exc.units, t_start=gsyn_exc.t_start,
                                               sampling_rate=gsyn_exc.sampling_rate)
        else:
            neuron_type = "all"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            for analogsignal in seg.analogsignals:
                if analogsignal.name == 'gsyn':
                    gsyn = analogsignal
        return gsyn
    
    
    def get_spike_trains(self, num_files=2):
        """
        Returns a list of neo.core.SpikeTrain elements representing the spike trains of all neurons, regardless
        of their type.
        Works only if there are two or one Pickle files containing the data (typically storing the excitatory and
        inhibitory data seperately, when there are two files).
        """
        self.set_directory_path()
        
        if num_files == 2:
            ### EXCITATORY NEURONS
            neuron_type = "exc"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            spiketrains_exc = seg.spiketrains
            
            ### INHIBITORY NEURONS
            neuron_type = "inh"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            spiketrains_inh = seg.spiketrains

            ### ALL NEURONS
            spiketrains = spiketrains_exc + spiketrains_inh
        else:
            neuron_type = "all"
            file_path   = self.get_file_path(neuron_type=neuron_type)
            PyNN_file   = open(file_path, "rb")
            loaded      = pickle.load(PyNN_file)
            seg         = loaded.segments[0] #we suppose there is only one segment
            spiketrains = seg.spiketrains
        return spiketrains

    
    def get_spike_train(self):
        global_spiketrain = list(chain.from_iterable(spiketrain for spiketrain in self.get_spike_trains()))
        global_spiketrain.sort()
        return global_spiketrain


    ###########################
    ### LFP related methods ###
    ###########################
    
    def produce_local_field_potential(self):
        """
        Calculates and returns the 1,2D-array of the LFP.
        The first dimension corresponds to the electrodes and does not exist if there is only one.
        """
        vm               = self.get_membrane_potential()
        gsyn             = self.get_conductance()
        neuron_positions = self.get_positions()

        num_neurons     = vm.shape[1] #equals to gsyn.shape[1]
        num_time_points = vm.shape[0]
        num_electrodes  = self.electrode_positions.shape[0]
        
        ones_array    = np.ones((num_electrodes, num_time_points, num_neurons))

        current_array = np.multiply(vm, gsyn)
        inv_dist      = nf.electrode_neuron_inv_dist(num_electrodes, num_neurons,
                                                     self.electrode_positions, neuron_positions,
                                                     self.reach, self.dimensionnality)

        big_current_array  = np.multiply(ones_array, current_array)
        big_inv_dist_array = np.multiply(ones_array, inv_dist)

        LFP = np.sum(big_current_array, big_inv_dist_array, axis=2)/(4*np.pi*self.sigma)

        if num_electrodes == 1:
            np.reshape(LFP, (num_time_points,))
        return LFP    


    def get_positions(self):
        """
        Returns the 2D-array giving the neurons' positions.
        """
        if self.space_dependency == False:
            positions = self.assign_positions()
        else:
            raise NotImplementedError("Must implement get_positions.")
        return positions


    def assign_positions(self):
        """
        Function that assigns positions to the neurons if they do not have.
        Only works if they have a 2D structure.
        """
        num_neurons = len(self.get_spike_trains)
        positions   = np.multiply(self.dimensions, np.random.rand(num_neurons, self.dimensionnality)-0.5)
        return positions
    

    ############################
    ### test related methods ###
    ############################
    
    def produce_vm_LFP_correlation(self, start=600, duration=1000, dt=0.1):
        """
        Calculates the correlation between the Vm of the closest neuron to the (first) electrode and the LFP signal
        recorded at this electrode.
        Returns the correlation and the corresponding lag (in ms).
        The relevant data is supposed to be 1s long.
        """
        start_index    = int(start/dt)
        duration_index = int(duration/dt)

        vm               = self.get_membrane_potential()
        neuron_positions = self.get_positions()

        num_neurons     = vm.shape[1]
        num_electrodes  = self.electrode_positions.shape[0]
        inv_dist        = nf.electrode_neuron_inv_dist(num_electrodes, num_neurons,
                                                       self.electrode_positions, neuron_positions,
                                                       self.reach, self.dimensionnality)
        closest_neuron = np.argmax(inv_dist)
        selected_vm    = np.reshape(vm[start_index:start_index+duration_index+1, closest_neuron], (duration_index+1,))
        selected_LFP   = np.reshape(self.produce_local_field_potential()[0, start_index:start_index+duration_index+1],
                                    (duration_index+1,))
        
        corr             = crsscorr.constwindowcorrelation(selected_vm, selected_LFP)
        corr_time_points = np.arange(-duration/2, duration/2+dt, dt)
        return corr, corr_time_points
    

    def produce_vm_LFP_coherence(self, start=600, duration=1000, dt=0.1):
        """
        Calculates the coherence between the Vm of the closest neuron to the (first) electrode and the LFP signal
        recorded at this electrode.
        returns the coherence and the corresponding frequencies (in Hz).
        The relevant data is supposed to be 1s long.
        """
        start_index    = int(start/dt)
        duration_index = int(duration/dt)

        vm               = self.get_membrane_potential()
        neuron_positions = self.get_positions()

        num_neurons     = vm.shape[1]
        num_electrodes  = self.electrode_positions.shape[0]
        inv_dist        = nf.electrode_neuron_inv_dist(num_electrodes, num_neurons,
                                                       self.electrode_positions, neuron_positions,
                                                       self.reach, self.dimensionnality)
        closest_neuron = np.argmax(inv_dist)
        selected_vm    = np.reshape(vm[start_index:start_index+duration_index+1, closest_neuron], (duration_index+1,))
        selected_LFP   = np.reshape(self.produce_local_field_potential()[0, start_index:start_index+duration_index+1],
                                    (duration_index+1,))

        f, coherenc = coherence(selected_vm, selected_LFP, fs=1000./dt)
        return coherenc, f
    

    def produce_phase_lock_value(self, start=525, offset=250, duration=1000, dt=0.1):
        """
        Calculates the Phase-Lock value for the spikes occuring in a 750ms period of time.
        The neurons are supposed to be excitated by a sinusoidal input of 1s, starting 250ms before the selected window.
        Returns the Phase-Lock value and the corresponding frequencies.
        """
        spiketrain = self.get_spike_train()
        num_spikes = len(spiketrain)
        valid_times1 = np.heaviside(spiketrain-(start+offset)*np.ones(num_spikes), 1)
        valid_times2 = np.heaviside((start+duration)*np.ones(num_spikes)-spiketrain, 1)
        valid_times  = np.multiply(valid_times1, valid_times2)

        selected_spikes = np.multiply(spiketrain, valid_times)
        selected_spikes = selected_spikes[selected_spikes>0]

        fs       = 1000./dt                                  #sampling frequency
        LFP      = self.produce_local_field_potential()
        #LFP_filt = butter_lowpass_filter(LFP, 170., fs)

        window       = 150                                   #ms, size of the window in which the LFP will have its Fourier transformations
        window_index = int(window/dt)
        window_width = window_index//2
        w            = hanning(window_index+1)               #150 ms window with a 0,1 ms interval
        N_max        = 500                                   #just an arbitrary value for the moment
        N_s = min(selected_spikes.shape[0], N_max)               #security measure
        PLv          = np.zeros(window_index+1, dtype=float) #Phase-Lock value array, empty for the moment

        for t_index in mf.random_list(N_s, selected_spikes.shape[0], minimum=0):
            t_s       = selected_spikes[t_index]                                         #time of spike occurence
            t_s_index = int(10*t_s)                                                      #corresponding index for the arrays
            #LFP_s     = LFP_filt[t_s_index - window_width : t_s_index + window_width+1]  #LFP centered at the spike occurrence
            LFP_s     = LFP[t_s_index - window_width : t_s_index + window_width+1]       #Non-filtered version
            wLFP_s    = np.multiply(w, LFP_s)                                            #centered LFP multiplied by the Hanning window
            FT_s      = fft(wLFP_s)                                                      #Fourier transform of this weighted LFP
            nFT_s     = np.divide(FT_s, np.abs(FT_s))                                    #normalized Fourier transform
            PLv       = np.add(PLv, nFT_s)                                               #contribution to the PLv added

        PLv  = np.abs(PLv)/N_s                                            #normalized module, according to the paper
        PLv  = PLv[:(PLv.shape[0])//2]                                    #only the first half is relevant
        fPLv = (0.5*fs/PLv.shape[0])*np.arange(PLv.shape[0], dtype=float) #frequencies of the PLv

        return PLv, fPLv