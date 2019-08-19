import numpy as np
from scipy.signal import butter, zpk2sos, sosfilt
from scipy.signal import lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    z,p,k = butter(order, normal_cutoff, output='zpk')
    return z,p,k

def butter_lowpass_filter(data, cutoff, fs, order=5):
    z,p,k = butter_lowpass(cutoff, fs, order)
    lesos = zpk2sos(z, p, k)
    return sosfilt(lesos, data)



def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)
        return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if 1 == 0:
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


    