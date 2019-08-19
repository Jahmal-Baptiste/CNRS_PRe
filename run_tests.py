from __future__ import absolute_import
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "Environnement_de_travail"

import sys, os
os.chdir("/media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail")
sys.path.append("/home/jahmal/.local/bin")
#for p in sys.path:
#    print(p)

import numpy as np
import matplotlib.pyplot as plt

from Validation.lfpmodels import CoulombModel


first_model = CoulombModel("First", network_model="T2")

LFP_bool = False
vm_bool  = True
gsyn_bool = False
corr_bool = False
coher_bool = False
stLFP_bool = False
PLv_bool   = False

if LFP_bool:
    LFP, time_points = first_model.produce_local_field_potential()
    LFP = np.reshape(LFP, (LFP.shape[1],))

    plt.figure()
    plt.title("First LFP")
    plt.plot(time_points, LFP)

if vm_bool:
    vm = first_model.get_membrane_potential()
    time_points = vm.times

    plt.figure()
    plt.title("First Vm")
    plt.plot(time_points, vm[:, 0])

if gsyn_bool:
    gsyn = first_model.get_conductance()

    plt.figure()
    plt.title("First Gsyn")
    plt.plot(time_points, gsyn[:, 0])


if corr_bool:
    zerolagcorrelations = first_model.produce_vm_LFP_zerolagcorrelations(trial_average=True)
    plt.figure()
    plt.hist(zerolagcorrelations)

if coher_bool:
    meancoherence, f, coherencestd = first_model.produce_vm_LFP_meancoherence(start=600)
    print(meancoherence.shape)
    print(coherencestd.shape)
    plt.figure()
    plt.semilogx(f, meancoherence)

if stLFP_bool:
    stLFP, window = first_model.produce_spike_triggered_LFP()
    plt.figure()
    plt.title("Spike-triggered LFP")
    plt.ylabel("LFP average (mV)")
    plt.xlabel("Lag (ms)")
    for k in range(stLFP.shape[0]):
        plt.plot(window, stLFP[k])

if PLv_bool:
    PLv, fPLv = first_model.produce_phase_lock_value()
    plt.figure()
    plt.title("Phase-Lock value")
    plt.xlabel("Frequency (Hz)")
    plt.plot(fPLv, PLv)
plt.show()