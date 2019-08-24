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
        corr[c]   = np.corrcoef(wa_d, wb)[0, 1]
        c += 1
    for d in range(window_size+1):
        wa        = a[:window_size]
        wb_d      = b[d:(d+window_size)]
        corr[c]   = np.corrcoef(wa, wb_d)[0, 1]
        c += 1
    return corr


def zerolagcorrelation(a, b):
    '''
    Normalized zero-lag cross-correlation between same-sized arrays a and b.
    '''
    return np.corrcoef(a, b)[0, 1]