from __future__ import absolute_import
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "Environnement_de_travail"

import numpy as np
import matplotlib.pyplot as plt

from .Validation.lfpmodels import CoulombModel


first_model = CoulombModel("First")

if 1==0:
    LFP, time_points = first_model.produce_local_field_potential()
    LFP = np.reshape(LFP, (LFP.shape[1],))

    plt.figure()
    plt.title("First LFP")
    plt.plot(time_points, LFP)

if 1==0:
    vm = first_model.get_membrane_potential()
    time_points = vm.times

    plt.figure()
    plt.title("First Vm")
    plt.plot(time_points, vm[:, 0])

if 1==0:
    gsyn = first_model.get_conductance()

    plt.figure()
    plt.title("First Gsyn")
    plt.plot(time_points, gsyn[:, 0])


if 1==0:
    zerolagcorrelations = first_model.produce_vm_LFP_zerolagcorrelations(trial_average=True)
    plt.figure()
    plt.hist(zerolagcorrelations)

if 1==1:
    meancoherence, f, coherencestd = first_model.produce_vm_LFP_meancoherence(start=600)
    print(meancoherence.shape)
    print(coherencestd.shape)
    plt.semilogx(f, meancoherence)



plt.show()