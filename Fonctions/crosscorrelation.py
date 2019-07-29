from scipy.signal import correlate, fftconvolve
from scipy.stats import wilcoxon
import numpy as np


def constwindowcorrelation(a, b):
    '''
    Normalized cross-correlation array with a fixed interval size for each time point of calculation.
    a and b are supposed to have the same shape.
    '''
    window_size = (a.shape[0])//2                                #we suppose that a and b are of same size

    corr = np.zeros(a.shape[0], dtype=float)                     #cross-correlation array
    c = 0                                                        #just the counter for the position of calculated correlation in the array
    for d in range(window_size, 0, -1):
        wa_d      = a[d:(d+window_size)]                         #wa_d for "windowed a of delay d"
        wb        = b[:window_size]                              #wb for "windowed b"
        raw_corr  = correlate(wa_d, wb, mode='valid')            #seen as a scalar product between wa_d and wb
        norm_wa_d = np.sqrt(correlate(wa_d, wa_d, mode='valid')) #seen as the norm of the scalar product defined above
        norm_wb   = np.sqrt(correlate(wb, wb, mode='valid'))     #seen as the norm of the scalar product defined above
        corr[c]   = raw_corr/(norm_wa_d*norm_wb)                 #cross-coreelation in interval [-1, 1]
        c += 1
    for d in range(window_size+1):
        wa        = a[:window_size]
        wb_d      = b[d:(d+window_size)]
        raw_corr  = correlate(wa, wb_d, mode='valid')
        norm_wa   = np.sqrt(correlate(wa, wa, mode='valid'))
        norm_wb_d = np.sqrt(correlate(wb_d, wb_d, mode='valid'))
        corr[c]   = raw_corr/(norm_wa*norm_wb_d)
        c += 1
    return corr


def constwilcoxcorrelation(a, b):
    '''Estimated cross-correlation array with a fixed interval size for each time point of calculation
    relying on the Wilcoxon test.'''
    window_size = (a.shape[0])//2               #we suppose that a and b are of same size

    corr = np.zeros(a.shape[0], dtype=float)    #cross-correlation array
    c = 0                                       #just the counter for the position of calculated correlation in the array
    for d in range(window_size, 0, -1):
        wa_d    = a[d:(d+window_size)]          #wa_d for "windowed a of delay d"
        wb      = b[:window_size]               #wb for "windowed b"
        corr[c] = wilcoxon(wa_d, wb)[0]         #estimate correlation
        c += 1
    for d in range(window_size+1):
        wa      = a[:window_size]
        wb_d    = b[d:(d+window_size)]
        corr[c] = wilcoxon(wa, wb_d)[0]
        c += 1
    return corr