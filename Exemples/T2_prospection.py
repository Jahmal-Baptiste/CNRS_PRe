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

import sys
sys.path.append("..")
from Fonctions import math_functions as mf


with open("../T2/ThalamoCorticalModel_data_size_____/Segment1.pickle", "rb") as PyNN_file:
    while True:
        try:
            plot = True
            seg = pickle.load(PyNN_file, encoding="latin1")
            print("EXCITATORY NEURONS\n")
            #print("Type of the seg file: " + str(type(seg)) + "\n") #this is a Segment...
            print("Segment description:\n\n" + seg.description)
            print("Segment annotations:\n\n" + str(seg.annotations) + "\n")


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


            print("")
            print("    Number of irregularlysampledsignals: " + str(len(seg.irregularlysampledsignals)) + "\n")
            print("    Number of spiketrains: " + str(len(seg.spiketrains)) + "\n")
            if plot:
                plt.figure()
                plt.title("Spike trains")
                rnd_list = mf.random_list(500, len(seg.spiketrains), minimum=0)
                for k in range(500):    
                    spiketrain_index = [k for i in range(len(seg.spiketrains[rnd_list[k]]))]
                    plt.scatter(seg.spiketrains[rnd_list[k]], spiketrain_index, marker="+")
            print("    Number of events: " + str(len(seg.events)) + "\n")
            print("    Number of epochs: " + str(len(seg.epochs)) + "\n")

            plt.show()
        except EOFError:
            break