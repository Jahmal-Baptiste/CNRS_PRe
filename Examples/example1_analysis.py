import numpy as np
import quantities as quant
import matplotlib.pyplot as plt
import neo
import pickle        


with open("./Results/20190705/current_injection_neuron_20190705-131959.pkl", "rb") as PyNN_file:
    while True:
        try:
            loaded = pickle.load(PyNN_file)
            #print("Type of the loaded file: " + str(type(loaded)) + "\n") #this is a Block
            print("Block description:\n" + loaded.description + "\n")
            print("Block annotations:\n" + str(loaded.annotations) + "\n")

            ################
            ### SEGMENTS ###
            ################

            #print("Number of segments: " + str(len(loaded.segments)) + "\n")
            seg = loaded.segments[0] #there is only one segment
            #print("Segment description:\n" + str(seg.description) + "\n")
            #print("Segment annotations:\n" + str(seg.annotations) + "\n")
            

            ##################################
            ### PROSPECTION OF THE SEGMENT ###
            ##################################

            #print("Number of analogsignals: " + str(len(seg.analogsignals)) + "\n")
            #print("Number of irregularlysampledsignals: " + str(len(seg.irregularlysampledsignals)) + "\n")
            #print("Number of spiketrains: " + str(len(seg.spiketrains)) + "\n")
            #print("Number of events: " + str(len(seg.events)) + "\n")
            #print("Number of epochs: " + str(len(seg.epochs)) + "\n")
            
            
            #####################
            ### ANALOGSIGNALS ###
            #####################
            
            asig = seg.analogsignals[0] #there is only one analogsignal
            print("Shape of AnalogSignal: " + str(asig.shape) + "\n") #there are 4 signals sampled in 5001 different times
            print("AnalogSignal units: " + str(asig.units) + "\n")


            ### PLOT OF ANALOGSIGNALS ###

            time_points = asig.times
            plt.figure()
            for k in range(4):
                plt.plot(time_points, asig[:, k])
            plt.show()
            plt.close()

        except EOFError:
            break