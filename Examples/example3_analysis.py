"""
This script analyses the data relative to the example 3 simulations' results.
The example 3 corresponds to an implementation of the Vogels Abbott model.

This script realises the following:
- plots the direct data readable from the pickle files containing the results;
- transforms the data into some relevant information (mean values, spike rates, interspike intervals etc.);
- computes the LFP signal using the Coulomb law's model (adding a spatial aspect to the neurons);
- realizes tests on the validity of the LFP's model inspired by the litterature (Vm-LFP correlation, coherence etc.).
"""

import pickle
import neo
import numpy as np
import quantities as quant
from scipy.signal import hanning, coherence
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from itertools import chain
import sys#, os, os.path as path
sys.path.append("..") #not the best way to modify sys.path
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__)))) # 2
#dossier = os.path.dirname(os.path.abspath(__file__))
#while not dossier.endswith('Fonctions'):
#    dossier = os.path.dirname(dossier)
#dossier = os.path.dirname(dossier)
#if dossier not in sys.path:
#    sys.path.append(dossier)
from Functions.filters import butter_lowpass_filter, butter_bandpass_filter
from Functions.crosscorrelation import constwindowcorrelation
from Functions.math_functions import random_list
from Functions.neuron_functions import electrode_neuron_inv_dist, spike_rate


#=================================================================================================================
#=================================================================================================================


reach = 0.001 #
network_width = 0.002 #m
dimensionnality = 2
electrode_position = np.array([[0., 0.]])
num_electrodes = electrode_position.shape[0]
siemens = 0.3*1e6#*quant.uS
sigma = siemens#/(1*quant.m)
pi_number = np.pi
fs = 10000. #sample rate in Hz


#=================================================================================================================
#=================================================================================================================


print("\nEXCITATORY NEURONS\n")
    
PyNN_file = open("./Results/20190718/VAbenchmarks_COBA_exc_neuron_np1_20190718-201157.pkl", "rb")
loaded    = pickle.load(PyNN_file)
seg       = loaded.segments[0] #there is only one segment

#####################
### ANALOGSIGNALS ###
#####################

print("Plots of the conductances...")
### SAMPLING OF CONDUCTANCES

for analogsignal in seg.analogsignals:
    if analogsignal.name == "gsyn_exc":
        gsyn_exc = analogsignal

n_plot      = 2
num_sig     = gsyn_exc.shape[1] #number of effective signals
rnd_list1   = random_list(n_plot, num_sig)
time_points = gsyn_exc.times
num_time_points = len(time_points)

plt.figure()
plt.subplot(3,1,1)
plt.title("Excitatory neurons")
plt.ylabel("Neurons' Gsyn (uS)")

for k in range(n_plot):
    plt.plot(time_points, gsyn_exc[:, rnd_list1[k]])

plt.plot(time_points, gsyn_exc[:, 0], color='r')


### PLOT OF THE AVERAGE CONDUCTANCE

avg_gsyn_exc = np.average(gsyn_exc, axis=1)
plt.plot(time_points, avg_gsyn_exc, color='k')


print("Plots of the Vms...")
### SAMPLING OF VMS ###

for analogsignal in seg.analogsignals:
    if analogsignal.name == 'v':
        vm_exc = analogsignal

n_plot = 2

plt.subplot(3,1,2)
plt.ylabel("Neurons' Vm (mV)")

for k in range(n_plot):
    plt.plot(time_points, vm_exc[:, rnd_list1[k]])

plt.plot(time_points, vm_exc[:, 0], color='r')

### PLOT OF THE AVERAGE VM

avg_vm_exc = np.average(vm_exc, axis=1)
plt.plot(time_points, avg_vm_exc, color='k')


###################
### SPIKETRAINS ###
###################

print("Plot of the spiketrains...")
### SAMPLING OF THE SPIKETRAINS

spiketrains_exc = seg.spiketrains

n_st      = min(150, num_sig)
rnd_list2 = random_list(n_st, num_sig, minimum=0)

plt.subplot(3,1,3)
plt.ylabel("Neurons' spike trains")
plt.xlabel("time (ms)")

for k in range(n_st):
    while len(spiketrains_exc[rnd_list2[k]]) == 0:
        rnd_list2[k]+=1 #Actually this algorithm is not good... But anyway
    
    spiketrain_index = [k for i in range(len(spiketrains_exc[rnd_list2[k]]))]
    plt.scatter(spiketrains_exc[rnd_list2[k]], spiketrain_index, marker="+")

if len(spiketrains_exc[0]) > 1:
    spiketrain_index = [n_st for i in range(len(spiketrains_exc[0]))]
    plt.scatter(spiketrains_exc[0], spiketrain_index, color='r', marker="+")


### PLOT OF THE INTERSPIKING INTERVALS

if 1 == 0:
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("Excitatory neurons")
    plt.ylabel("Reference neuron ISI")
    ref_intervals = np.diff(spiketrains_exc[0])
    ref_intervals = list(map(int, ref_intervals))
    plt.hist(ref_intervals, bins=50)

    plt.subplot(2,1,2)
    plt.ylabel("Random neuron ISI")
    rnd_num = np.random.random_integers(1, num_sig-1)
    rnd_intervals = np.diff(spiketrains_exc[rnd_num])
    rnd_intervals = list(map(int, rnd_intervals))
    plt.hist(rnd_intervals, bins=50)


### PLOT OF THE SPIKE RATES

"""
The spike rates will be plotted three histograms (one before the stimulus, one during and one after).
The histograms will give the number of neurons that have a spike rate in a given interval.

To have a clean code I will have to store and use the stimulus information in an epoch object.
"""
if 1 == 0:
        print('Plot of the spike rates...')
        plt.figure()
        plt.subplot(1,3,1)
        plt.title("Excitatory neurons")
        plt.ylabel("Spiking rate")

        pre_stim_spikerates = []
        beginning = 400.
        ending    = 450.
        for index in range(num_sig):
                pre_stim_spikerates.append(spike_rate(spiketrains_exc[index], 0., beginning)) #I will have to change this line/section to take into consideration an epoch specifying the stimulus
        plt.hist(pre_stim_spikerates, bins=30)

        plt.subplot(1,3,2)
        stim_spikerates = []
        for index in range(num_sig):
                stim_spikerates.append(spike_rate(spiketrains_exc[index], beginning, ending))
        plt.hist(stim_spikerates, bins=30)

        plt.subplot(1,3,3)
        post_stim_spikerates = []
        for index in range(num_sig):
                post_stim_spikerates.append(spike_rate(spiketrains_exc[index], ending, 1600.))
        plt.hist(post_stim_spikerates, bins=30)


### PLOT OF THE SPIKE OCCURENCES
"""
Plot of the number of spikes in th given bins.
"""
if 1 == 0:
        print('Plot of the spike occurences...')
        bin_size      = 10 #10 ms interval
        s_time_points = [bin_size*k for k in range(1600//bin_size)]
        s_time_points.append(1600)
        s_heights     = [0 for k in range(len(s_time_points))]

        for spiketrain in spiketrains_exc:
            for t in spiketrain:
                s_heights[int(t/bin_size)] += 1 #each t is a float "accurate to the decimal"

        plt.figure()
        plt.title("Excitatory neurons")
        plt.xlabel("Time intervals (ms)")
        plt.ylabel("Number of neurons")
        plt.scatter(s_time_points, s_heights, marker="x")


###########
### LFP ###
###########

print('Computation of the distances...')
positions_exc = network_width*(np.random.rand(num_sig, dimensionnality)-0.5)
inv_distances_exc = electrode_neuron_inv_dist(num_electrodes, num_sig, electrode_position, positions_exc, reach, dimensionnality)[0, :]

print('Computation of the excitatory LFP...')
current_array = np.multiply(vm_exc, gsyn_exc)
indiv_LFP     = np.multiply(current_array, inv_distances_exc)#*(quant.mV)*(quant.uS)/quant.m
LFP_exc       = np.sum(indiv_LFP, axis=1)/(4*np.pi*sigma)


#=================================================================================================================
#=================================================================================================================


print("\nINHIBITORY NEURONS\n")

PyNN_file = open("./Results/20190718/VAbenchmarks_COBA_inh_neuron_np1_20190718-201157.pkl", "rb")
loaded    = pickle.load(PyNN_file)
seg       = loaded.segments[0] #there is only one segment

#####################
### ANALOGSIGNALS ###
#####################

print('Plots of the conductances...')
### SAMPLING OF CONDUCTANCES

for analogsignal in seg.analogsignals:
    if analogsignal.name == "gsyn_inh":
        gsyn_inh = analogsignal

n_plot    = 3
num_sig   = gsyn_inh.shape[1] #number of effective signals
rnd_list1 = random_list(n_plot, num_sig)

plt.figure()
plt.subplot(3,1,1)
plt.title("Inhibitory neurons")
plt.ylabel("Neurons' Gsyn (uS)")

for k in range(n_plot):
    plt.plot(time_points, gsyn_inh[:, rnd_list1[k]])

### PLOT OF THE AVERAGE CONDUCTANCE

avg_gsyn_inh = np.average(gsyn_inh, axis=1)
plt.plot(time_points, avg_gsyn_inh, color='k')


print('Plots of the Vms...')
### SAMPLING OF VMS

for analogsignal in seg.analogsignals:
    if analogsignal.name == "v":
        vm_inh = analogsignal

plt.subplot(3,1,2)
plt.ylabel("Neurons' Vm (mV)")
for k in range(n_plot):
    plt.plot(time_points, vm_inh[:, rnd_list1[k]])

### PLOT OF THE AVERAGE VM

avg_vm_inh = np.average(vm_inh, axis=1)
plt.plot(time_points, avg_vm_inh, color='k')


###################
### SPIKETRAINS ###
###################

print('Plot of the spiketrains...')
### SAMPLING OF SPIKETRAINS

n_st      = min(150, num_sig)
rnd_list2 = random_list(n_st, num_sig, minimum=0)
plt.subplot(3,1,3)
plt.ylabel("Neurons' spike trains")
plt.xlabel("time (ms)")

spiketrains_inh = seg.spiketrains

for k in range(n_st):
    while len(spiketrains_inh[rnd_list2[k]]) == 0:
        rnd_list2[k]+=1

    #spiketrain = [rnd_list2[k] for i in range(len(spiketrains_inh[rnd_list2[k]]))]
    spiketrain = [k for i in range(len(spiketrains_inh[rnd_list2[k]]))]
    plt.scatter(spiketrains_inh[rnd_list2[k]], spiketrain, marker="+")


### PLOT OF THE INTERSPIKING INTERVALS

#plt.figure()
#plt.title("Inhibitory neurons")
#plt.ylabel("Random neuron's ISI")
#rnd_num = np.random.random_integers(0, num_sig-1)
#rnd_intervals = np.diff(spiketrains_inh[rnd_num])
#rnd_intervals = list(map(int, rnd_intervals))
#plt.hist(rnd_intervals, bins=30)


### PLOT OF THE SPIKE RATES

"""
The spike rates will be plotted three histograms (one before the stimulus, one during and one after).
The histograms will give the number of neurons that have a spike rate in a given interval.
"""
if 1 == 0:
        print('Plot of the spike rates...')
        plt.figure()
        plt.subplot(1,3,1)
        plt.title("Inhibitory neurons")
        plt.ylabel("Spiking rate")
        pre_stim_spikerates = []
        beginning = 400.
        ending    = 450.
        for index in range(num_sig):
                pre_stim_spikerates.append(spike_rate(spiketrains_inh[index], 0., beginning)) #I will have to change this line/section to take into consideration an epoch specifying the stimulus
        plt.hist(pre_stim_spikerates, bins=30)

        plt.subplot(1,3,2)
        stim_spikerates = []
        for index in range(num_sig):
                stim_spikerates.append(spike_rate(spiketrains_inh[index], beginning, ending))
        plt.hist(stim_spikerates, bins=30)

        plt.subplot(1,3,3)
        post_stim_spikerates = []
        for index in range(num_sig):
                post_stim_spikerates.append(spike_rate(spiketrains_inh[index], ending, 1600.))
        plt.hist(post_stim_spikerates, bins=30)


### PLOT OF THE SPIKE OCCURENCES
"""
Plot of the number of spikes in th given bins.
"""
if 1 == 0:
        print('Plot of the spike occurences...')
        bin_size      = 10 #10 ms interval
        s_time_points = [bin_size*k for k in range(1600//bin_size)]
        s_time_points.append(1600)
        s_heights     = [0 for k in range(len(s_time_points))]

        for spiketrain in spiketrains_inh:
            for t in spiketrain:
                s_heights[int(t/bin_size)] += 1 #each t is a float "accurate to the decimal"

        plt.figure()
        plt.title("Inhibitory neurons")
        plt.xlabel("Time intervals (ms)")
        plt.ylabel("Number of neurons")
        plt.scatter(s_time_points, s_heights, marker="x")


###########
### LFP ###
###########

print('Computation of the distances...')
positions_inh = network_width*(np.random.rand(num_sig, dimensionnality)-0.5)
inv_distances_inh = electrode_neuron_inv_dist(num_electrodes, num_sig, electrode_position, positions_inh, reach, dimensionnality)[0, :]

print('Computation of the inhibitory LFP...')
current_array = np.multiply(vm_inh, gsyn_inh)
indiv_LFP     = np.multiply(current_array, inv_distances_inh)
LFP_inh       = np.sum(indiv_LFP, axis=1)/(4*np.pi*sigma)


#=================================================================================================================
#=================================================================================================================


print('\nComputation of the total LFP...')
LFP_tot = np.add(LFP_exc, LFP_inh)

print('Plot of the LFP signals...')
plt.figure()
plt.title("LFP signals")
plt.ylabel("LFP (mV)")
plt.xlabel("time (ms)")
plt.plot(time_points, LFP_exc, label='Excitatory component')
plt.plot(time_points, LFP_inh, label='Inhibitory component')
plt.plot(time_points, LFP_tot, color='r', label='Total Field')
plt.legend()


#=================================================================================================================
#=================================================================================================================

print('Computation of the filtered signals...')
'''
I am wondering how to get rid of the contradiction that lays in the sampling frequency that remains high (10 kHz)
while I work with filtered (or not, arcording to how good the filter works...) signals at low frequencies (50/170 Hz).
I can't pretend that my sampling frequency is not high I guess...
'''

closest_exc_index = np.argmax(inv_distances_exc)
#closest_inh_index = np.argmax(inv_distances_inh)
closest_vm_exc    = vm_exc[:, closest_exc_index]
#closest_vm_inh    = vm_inh[:, closest_inh_index]
#vm_exc_filt1      = butter_lowpass_filter(closest_vm_exc, 50., fs, order=15) #does not filter anything...
#LFP_filt1         = butter_lowpass_filter(LFP_tot,        50., fs)
LFP_filt2         = butter_lowpass_filter(LFP_tot,       170., fs)
#vm_exc_filt1      = butter_bandpass_filter(closest_vm_exc, 0.5, 50.,  fs, order=9)
#LFP_filt1         = butter_bandpass_filter(LFP_tot,      0.5, 50.,  fs)    #bandpass filtered LFP between 0.5 and 50 Hz
#LFP_filt2         = butter_bandpass_filter(LFP_tot,      0.7, 170., fs)   #bandpass filtered LFP between 0.7 and 170 Hz

plt.figure()
plt.subplot(2,1,1)
plt.title("Local Field Potential")
plt.ylabel("LFP (mV)")
plt.xlabel("time (ms)")
plt.plot(time_points, LFP_tot, '--', label='Non-filtered Field')
#plt.plot(time_points, LFP_filt1,     label='Filtered Field at <50 Hz')
plt.plot(time_points, LFP_filt2,     label='Filtered Field at <170 Hz')
plt.legend()

plt.subplot(2,1,2)
plt.title("Membrane potential")
plt.ylabel("Vm (mV)")
plt.xlabel("time (ms)")
plt.plot(time_points, closest_vm_exc, label='Non-filtered Vm')
#plt.plot(time_points, vm_exc_filt1,   label='Filtered Vm at <50 Hz')
plt.legend()



#=================================================================================================================
#=================================================================================================================

print('\nTEST PHASE')

### VM-LFP COREELATION AND COHERENCE

print('\nCorrelation and coherence...')

#selected_LFP = np.array(LFP_filt1[6000:])                #filtered at <50 Hz
#selected_vm  = np.reshape(vm_exc_filt1[6000:], (10001,)) #filtered at <50 Hz
selected_LFP = np.array(LFP_tot[6000:])                  #Non-filtered
selected_vm  = np.reshape(closest_vm_exc[6000:], (10001,))         #Non-filtered

corr             = constwindowcorrelation(selected_vm, selected_LFP)
corr_time_points = np.arange(-500, 500.1, 0.1)
plt.figure()
plt.title("Vm-LFP Cross-correlation")
plt.ylabel("Correlation coefficient")
plt.xlabel("Lag (ms)")
plt.plot(corr_time_points, corr, label='constwindowcorrelation')
plt.legend()

plt.figure()
f, coherenc = coherence(selected_vm, selected_LFP, nperseg=int(2**12), noverlap=None, fs=fs)
print(f.shape)
plt.title("Vm-LFP Coherence")
plt.ylabel("Coherence")
plt.xlabel("Frequency (Hz)")
plt.semilogx(f[:205], coherenc[:205], label='Half-overlap')
plt.legend()



### PHASE-LOCK VALUE

if 1 == 0:
    print('\nPhase-lock value...')
    start = 525
    offset = 250
    duration = 1000
    window       = 150                                   #ms, size of the window in which the LFP will have its Fourier transformations
    window_index = 10*window                             #equals to window/0.1
    window_width = window_index//2
    w            = hanning(window_index+1)               #150 ms window with a 0,1 ms interval
    N_max        = 500                                   #just an arbitrary value for the moment
    PLv          = np.zeros(window_index+1, dtype=float) #Phase-Lock value array, empty for the moment
    #PLv_list     = []                                   #useless for the moment because serves when many trials are done (many segments)

    global_spiketrains = spiketrains_exc + spiketrains_inh
    global_spiketrain  = list(chain.from_iterable(spiketrain for spiketrain in global_spiketrains))
    #print('Number of spikes: ' + str(len(global_spiketrain)))

    num_spikes = len(global_spiketrain)
    valid_times1 = np.heaviside(global_spiketrain-(start+offset)*np.ones(num_spikes), 1)
    valid_times2 = np.heaviside((start+duration)*np.ones(num_spikes)-global_spiketrain, 1)
    valid_times  = np.multiply(valid_times1, valid_times2)

    selected_spikes = np.multiply(global_spiketrain, valid_times)
    selected_spikes = selected_spikes[selected_spikes>0]
    np.sort(selected_spikes)

    #print('Number of selected spikes: ' + str(selected_spikes.shape[0]))

    N_s = min(selected_spikes.shape[0], N_max) #security measure

    for t_index in random_list(N_s, selected_spikes.shape[0], minimum=0):
        t_s       = selected_spikes[t_index]                                         #time of spike occurence
        t_s_index = int(10*t_s)                                                      #corresponding index for the arrays
        LFP_s     = LFP_filt2[t_s_index - window_width : t_s_index + window_width+1] #LFP centered at the spike occurrence
        LFP_s     = LFP_tot[t_s_index - window_width : t_s_index + window_width+1]   #Non-filtered version
        wLFP_s    = np.multiply(w, LFP_s)                                            #centered LFP multiplied by the Hanning window
        FT_s      = fft(wLFP_s)                                                      #Fourier transform of this weighted LFP
        nFT_s     = np.divide(FT_s, np.abs(FT_s))                                    #normalized Fourier transform
        PLv       = np.add(PLv, nFT_s)                                               #contribution to the PLv added

    PLv  = np.abs(PLv)/N_s                                            #normalized module, according to the paper
    PLv  = PLv[:(PLv.shape[0])//2]                                    #only the first half is relevant
    fPLv = (0.5*fs/PLv.shape[0])*np.arange(PLv.shape[0], dtype=float) #frequencies of the PLv

    plt.figure()
    plt.title('Phase-Lock value')
    plt.xlabel('frequency (Hz)')
    plt.plot(fPLv, PLv)

    #average_PLv = np.average(PLv_list) #useful only when PLv_list exists...


#=================================================================================================================
#=================================================================================================================


plt.show()
plt.close()