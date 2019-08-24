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
from  .Functions import math_functions as mf


first_model = CoulombModel("First", network_model="VA", space_dependency=False)

LFP_bool   = False
vm_bool    = False
gsyn_bool  = False
corr_bool  = False
coher_bool = False
PLv_bool   = True
stLFP_bool = False

trial_average = False

if LFP_bool:
    LFP, time_points = first_model.produce_local_field_potential(trial=0)
    LFP = np.reshape(LFP, (LFP.shape[1],))

    plt.figure()
    plt.title("LFP")
    plt.plot(time_points, LFP)

if vm_bool:
    vm = first_model.get_membrane_potential()
    time_points = vm.times

    plt.figure()
    plt.title("Vm")
    avg_vm = np.average(vm, axis=1)
    plt.plot(time_points, avg_vm, color='k')
    
    rnd_list = mf.random_list(2, vm.shape[1], minimum=0)
    for k in range(2):
        plt.plot(time_points, vm[:, rnd_list[k]])

if gsyn_bool:
    gsyn = first_model.get_conductance()
    time_points = gsyn.times

    plt.figure()
    plt.title("Gsyn")
    avg_gsyn = np.average(gsyn, axis=1)
    plt.plot(time_points, avg_gsyn, color='k')

    rnd_list = mf.random_list(2, vm.shape[1], minimum=0)
    for k in range(2):
        plt.plot(time_points, gsyn[:, rnd_list[k]])


if corr_bool:
    if first_model.network_model == "VA":
        start = 600
    else:
        start = 0
    #corr, corr_time_points = first_model.produce_vm_LFP_correlation(start=start)
    #plt.figure()
    #plt.title("Closest neuron's Vm-LFP correlation")
    #plt.xlabel("Lag (ms)")
    #plt.plot(corr_time_points, corr)
    
    zerolagcorrelations = first_model.produce_vm_LFP_zerolagcorrelations(trial_average=trial_average, withinreach=True)
    plt.figure()
    plt.title("Zerolag correlations")
    plt.xlabel("Correlation")
    plt.hist(zerolagcorrelations)

if coher_bool:
    if first_model.network_model == "VA":
        start = 600
    else:
        start = 0
    meancoherence, f, coherencestd = first_model.produce_vm_LFP_meancoherence(start=start, withinreach=True)
    plt.figure()
    plt.title("Mean coherence")
    plt.xlabel("Frequency (Hz)")
    plt.semilogx(f, meancoherence, color='steelblue')
    plt.fill_between(f, meancoherence-coherencestd, meancoherence+coherencestd, color='lightsteelblue')
    plt.xlim(0.5, 100)

if PLv_bool:
    if first_model.network_model == "VA":
        start = 600
    else:
        start = 0
    PLv, fPLv = first_model.produce_phase_lock_value(start=start, trial_average=trial_average, withinreach=True)
    plt.figure()
    plt.title("Phase-Lock value")
    plt.xlabel("Frequency (Hz)")
    plt.plot(fPLv, PLv)

if stLFP_bool:
    discrim_dist_list = [4e-4, 1.6e-3, 2.4e-3, 4e-3]
    stLFP, window = first_model.produce_spike_triggered_LFP(discrim_dist_list=discrim_dist_list, window_width=400)
    plt.figure()
    plt.title("Spike-triggered LFP")
    plt.ylabel("LFP average (mV)")
    plt.xlabel("Lag (ms)")
    for k in range(len(discrim_dist_list)):
        if stLFP[k, :].all() != 0:
            if k == 0:
                plt.plot(window, stLFP[k], label="d <= " + str(discrim_dist_list[0]*1e3))
            elif k == len(discrim_dist_list)-1:
                plt.plot(window, stLFP[k], label="d > " + str(discrim_dist_list[-1]*1e3))
            else:
                plt.plot(window, stLFP[k], label=str(discrim_dist_list[k]*1e3) + " < d <= " + str(discrim_dist_list[k+1]*1e3))
    plt.legend()

plt.show()