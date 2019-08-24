from __future__ import absolute_import
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "Environnement_de_travail"

"""
This script serves as a first-hand prospecter for the pickle files containing the T2 simulations' results.

This script prints the composition of each registered file corresponding to a given simulation: description,
annotations, number of segments, type of data registered etc.
"""

import numpy as np
import quantities as quant
import matplotlib.pyplot as plt
import neo
import pickle
import json

#import sys
#sys.path.append("..")
from Functions import math_functions as mf

content = True
positions = False
for k in range(1):
    if content:
        #with open("./T2/ThalamoCorticalModel_data_size_____/Segment{}.pickle".format(str(k)), "rb") as PyNN_file:
        with open("./T2/ThalamoCorticalModel_data_size_____/Segment11(1).pickle", "rb") as PyNN_file:
            while True:
                try:
                    plot = True
                    seg = pickle.load(PyNN_file, encoding="latin1")
                    #print("Type of the seg file: " + str(type(seg)) + "\n") #this is a Segment...
                    #print("Segment description:\n\n" + seg.description)
                    #print("Segment annotations:\n\n" + str(seg.annotations) + "\n")
                    #if 'DriftingSinusoidalGratingDisk' in seg.annotations['stimulus']:
                    #    if seg.annotations['sheet_name'] in ['V1_Exc_L4', 'V1_Inh_L4']:
                    #        print("Segment" + str(k) + ":")
                    #        print(seg.annotations)


                    ##################################
                    ### PROSPECTION OF THE SEGMENT ###
                    ##################################

                    num_analogsignals = len(seg.analogsignals)
                    print("    Number of analogsignals: " + str(num_analogsignals))
                    for k in range(num_analogsignals):
                        print("        " + str(seg.analogsignals[k].name) + " in " + str(seg.analogsignals[k].units))
                        print("        Shape: " + str(seg.analogsignals[k].shape))
                        if plot:
                            plt.figure()
                            plt.title(str(seg.analogsignals[k].name))
                            time_points = seg.analogsignals[k].times
                            for i in range(3):
                                plt.plot(time_points, seg.analogsignals[k][:, i])


                    #print("")
                    #print("    Number of irregularlysampledsignals: " + str(len(seg.irregularlysampledsignals)) + "\n")
                    print("    Number of spiketrains: " + str(len(seg.spiketrains)) + "\n")
                    if plot:
                        plt.figure()
                        plt.title("Spike trains of segment " + str(k) + " and sheet " + seg.annotations['sheet_name'])
                        rnd_list = mf.random_list(len(seg.spiketrains), len(seg.spiketrains), minimum=0)
                        for k in range(len(seg.spiketrains)):    
                            spiketrain_index = [k for i in range(len(seg.spiketrains[rnd_list[k]]))]
                            plt.scatter(seg.spiketrains[rnd_list[k]], spiketrain_index, marker="+")
                    print("    Number of events: " + str(len(seg.events)) + "\n")
                    print("    Number of epochs: " + str(len(seg.epochs)) + "\n")

                    plt.show()
                except EOFError:
                    break

if positions:
    with open("./T2/ThalamoCorticalModel_data_size_____/id_positions_V1_Inh_L4.pickle", "rb") as f:
        while True:
            try:
                positions = pickle.load(f, encoding="latin1")
                print(len(list(positions.keys())))
            except EOFError:
                break