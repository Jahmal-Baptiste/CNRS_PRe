from __future__ import absolute_import
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "Environnement_de_travail"

import sys, os
os.chdir("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail")
sys.path.append("/home/jahmal/.local/bin")


import numpy as np
import matplotlib.pyplot as plt

from .Validation.lfpmodels import CoulombModel
from .Functions import math_functions as mf
from .Functions import neuron_functions as nf

models        = ["VA", "T2"]
model_int     = 0
first_model   = CoulombModel("First", network_model=models[model_int], space_dependency=False)
trial_average = False

LFP_bool   = True
vm_bool    = True
gsyn_bool  = True
corr_bool  = True
coher_bool = True
PLv_bool   = True
stLFP_bool = True

save       = True

######################################################################################################################
if LFP_bool:
    single_plot = False
    if single_plot:
        LFP, time_points = first_model.produce_local_field_potential(trial=0)
        LFP = np.reshape(LFP, (LFP.shape[1],))

        plt.figure()
        plt.title("LFP")
        plt.xlabel("Time (ms)")
        plt.ylabel("Field (mV)")
        plt.plot(time_points, LFP)

    else:
        electrode_positions = nf.electrode_grid(9, first_model.dimensions, first_model.reach)
        first_model.electrode_positions = np.transpose(electrode_positions)
        num_electrodes = 9

        print("      At electrode number (0/{})...".format(num_electrodes-1))
        print(first_model.electrode_positions[0])
        first_LFP, time_points = first_model.produce_local_field_potential(electrode=0)
        first_LFP = np.reshape(first_LFP, (first_LFP.shape[1]))
        LFP       = np.zeros((num_electrodes, first_LFP.shape[0]))
        LFP[0]    = first_LFP
        count     = 1
        for electrode in range(1, num_electrodes):
            print("      At electrode number ({0}/{1})...".format(count, num_electrodes-1))
            print(first_model.electrode_positions[electrode])
            LFP[electrode] = first_model.produce_local_field_potential(electrode=electrode)[0][0, :]
            count         += 1
        
        plt.figure()
        plt.title("LFP")
        plt.xlabel("Time (ms)")
        plt.ylabel("Field (mV)")
        for k in range(num_electrodes-1):
            plt.plot(time_points, LFP[k])
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_LFP.svg".format(models[model_int]),
        format="svg")


######################################################################################################################
if vm_bool:
    vm = first_model.get_membrane_potential()
    time_points = vm.times

    plt.figure()
    plt.title("Vm")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    avg_vm = np.average(vm, axis=1)
    plt.plot(time_points, avg_vm, color='k', label='Average Vm')
    
    rnd_list = mf.random_list(1, vm.shape[1], minimum=0)
    for k in range(1):
        plt.plot(time_points, vm[:, rnd_list[k]], label="Random neuron's Vm")
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_Vm.svg".format(models[model_int]),
        format="svg")


######################################################################################################################
if gsyn_bool:
    gsyn = first_model.get_conductance()
    time_points = gsyn.times

    plt.figure()
    plt.title("Gsyn")
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductance (uS)")
    avg_gsyn = np.average(gsyn, axis=1)
    plt.plot(time_points, avg_gsyn, color='k', label="Average Conductance")

    rnd_list = mf.random_list(1, vm.shape[1], minimum=0)
    for k in range(1):
        plt.plot(time_points, gsyn[:, rnd_list[k]], label="Random neuron's Conductance")
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_Gsyn.svg".format(models[model_int]),
        format="svg")


######################################################################################################################
if corr_bool:
    if first_model.network_model == "VA":
        start       = 600
    else:
        start       = 0
    if first_model.experiment == 'sin_stim':
        color_index = 1
    else:
        color_index = 0
    deepcolors  = ['seagreen',     'steelblue']
    lightcolors = ['darkseagreen', 'lightsteelblue']

    #corr, corr_time_points = first_model.produce_vm_LFP_correlation(start=start)
    #plt.figure()
    #plt.title("Closest neuron's Vm-LFP correlation")
    #plt.xlabel("Lag (ms)")
    #plt.plot(corr_time_points, corr)
    
    zerolagcorrelations = first_model.produce_vm_LFP_zerolagcorrelations(trial_average=trial_average, withinreach=True)
    plt.figure()
    plt.title("Zerolag correlations")
    plt.xlabel("Correlation magnitude")
    plt.ylabel("Neuron density")
    plt.hist(zerolagcorrelations, density=True, bins=13, color=deepcolors[color_index])
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_Corr.svg".format(models[model_int]),
        format="svg")


######################################################################################################################
if coher_bool:
    if first_model.network_model == "VA":
        start       = 600
    else:
        start       = 0
    if first_model.experiment == 'sin_stim':
        color_index = 1
    else:
        color_index = 0
    deepcolors  = ['seagreen',     'steelblue']
    lightcolors = ['darkseagreen', 'lightsteelblue']

    meancoherence, f, coherencestd = first_model.produce_vm_LFP_meancoherence(start=start, withinreach=True)
    plt.figure()
    plt.title("Mean coherence")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence magnitude")
    plt.semilogx(f, meancoherence, color=deepcolors[color_index])
    plt.fill_between(f, meancoherence-coherencestd, meancoherence+coherencestd, color=lightcolors[color_index])
    plt.xlim(0.5, 100)
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_Coher.svg".format(models[model_int]),
        format="svg")


######################################################################################################################
if PLv_bool:
    electrode_positions = nf.electrode_grid(9, first_model.dimensions, first_model.reach)
    first_model.electrode_positions = np.transpose(electrode_positions)
    if first_model.network_model == "VA":
        start = 600
    else:
        start = 0
    PLv, fPLv, PLv_std = first_model.produce_phase_lock_value(start=start, trial_average=trial_average, withinreach=True)
    plt.figure()
    plt.title("Phase-Lock value")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spike-Field Phase-Locking Value")
    plt.plot(fPLv, PLv, color='k')
    plt.fill_between(fPLv, PLv-PLv_std, PLv+PLv_std, color='gray')
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_PLv.svg".format(models[model_int]),
        format="svg")


######################################################################################################################
if stLFP_bool:
    discrim_dist_list = [4e-4, 1.6e-3, 2.4e-3, 4e-3]
    colors            = ['r', 'dodgerblue', 'deepskyblue', 'yellowgreen']
    stLFP, window     = first_model.produce_spike_triggered_LFP(discrim_dist_list=discrim_dist_list, window_width=400)
    plt.figure()
    plt.title("Spike-triggered LFP")
    plt.ylabel("LFP average (mV)")
    plt.xlabel("Lag (ms)")
    for k in range(len(discrim_dist_list)):
        if stLFP[k, :].all() != 0:
            if k == 0:
                plt.plot(window, stLFP[k],
                         label="d <= " + str(discrim_dist_list[0]*1e3),
                         color=colors[k])
            elif k == len(discrim_dist_list)-1:
                plt.plot(window, stLFP[k],
                         label="d > " + str(discrim_dist_list[-1]*1e3),
                         color=colors[k])
            else:
                plt.plot(window, stLFP[k],
                         label=str(discrim_dist_list[k]*1e3) + " < d <= " + str(discrim_dist_list[k+1]*1e3),
                         color=colors[k])
    plt.legend()
    if save:
        plt.savefig("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Results/{}/{}_stLFP.svg".format(models[model_int]),
        format="svg")

plt.show()