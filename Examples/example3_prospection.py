"""
This script serves as a first-hand prospecter for the pickle files containing the example 3 simulations' results.
The example 3 is a code implementing the Voggels Abbott model.

Each simulation is supposed to have recorded its data into two pickle files:
- one for the excitatory neurons;
- one for the inhibitory neurons.

This script prints the composition of each registered file corresponding to a given simulation: description,
annotations, number of segments, type of data registered etc.
"""

import numpy as np
import quantities as quant
import matplotlib.pyplot as plt
import neo
import pickle


with open("./Results/20190710/VAbenchmarks_COBA_exc_neuron_np1_20190710-124149.pkl", "rb") as PyNN_file:
    while True:
        try:
            loaded = pickle.load(PyNN_file)
            print("EXCITATORY NEURONS\n")
            #print("Type of the loaded file: " + str(type(loaded)) + "\n") #this is a Block
            print("Block description:\n" + loaded.description)
            print("Block annotations:\n" + str(loaded.annotations) + "\n")

            ################
            ### SEGMENTS ###
            ################

            print("Number of segments: " + str(len(loaded.segments)) + "\n")


            ##################################
            ### PROSPECTION OF THE SEGMENT ###
            ##################################

            for seg in loaded.segments:
                num_analogsignals = len(seg.analogsignals)
                print("    Number of analogsignals: " + str(num_analogsignals))
                for k in range(num_analogsignals):
                    print("        " + str(seg.analogsignals[k].name) + " in " + str(seg.analogsignals[k].units))
                print("")
                print("    Number of irregularlysampledsignals: " + str(len(seg.irregularlysampledsignals)) + "\n")
                print("    Number of spiketrains: " + str(len(seg.spiketrains)) + "\n")
                print("    Number of events: " + str(len(seg.events)) + "\n")
                print("    Number of epochs: " + str(len(seg.epochs)) + "\n")


        except EOFError:
            break

with open("./Results/20190710/VAbenchmarks_COBA_inh_neuron_np1_20190710-124149.pkl", "rb") as PyNN_file:
    while True:
        try:
            loaded = pickle.load(PyNN_file)
            print("\nINHIBITORY NEURONS\n")
            #print("Type of the loaded file: " + str(type(loaded)) + "\n") #this is a Block
            print("Block description:\n" + loaded.description)
            print("Block annotations:\n" + str(loaded.annotations) + "\n")

            ################
            ### SEGMENTS ###
            ################

            print("Number of segments: " + str(len(loaded.segments)) + "\n")

            ##################################
            ### PROSPECTION OF THE SEGMENT ###
            ##################################

            for seg in loaded.segments:
                num_analogsignals = len(seg.analogsignals)
                print("    Number of analogsignals: " + str(num_analogsignals))
                for k in range(num_analogsignals):
                    print("        " + str(seg.analogsignals[k].name) + " in " + str(seg.analogsignals[k].units))
                print("")
                print("    Number of irregularlysampledsignals: " + str(len(seg.irregularlysampledsignals)) + "\n")
                print("    Number of spiketrains: " + str(len(seg.spiketrains)) + "\n")
                print("    Number of events: " + str(len(seg.events)) + "\n")
                print("    Number of epochs: " + str(len(seg.epochs)) + "\n")

        except EOFError:
            break