import numpy as np
import matplotlib.pyplot as plt

from Validation.lfpmodels import CoulombModel


first_model = CoulombModel("First")

#LFP, time_points = first_model.produce_local_field_potential()
#LFP = np.reshape(LFP, (LFP.shape[1],))

#plt.figure()
#plt.title("First LFP")
#plt.plot(time_points, LFP)


vm = first_model.get_membrane_potential()
time_points = vm.times

plt.figure()
plt.title("First Vm")
plt.plot(time_points, vm[:, 0])

gsyn = first_model.get_conductance()

plt.figure()
plt.title("First Gsyn")
plt.plot(time_points, gsyn[:, 0])

plt.show()