import numpy as np


def random_list(n, maximum, minimum=1):
    '''Returns a list of n random numbers between 1 (included) and maximum (excluded), all different.'''
    if n >= maximum-1:
        return [range(minimum, maximum)]
    rnd_list = []
    k = 0
    while k<n:
        rnd_num = np.random.random_integers(minimum, maximum-1)
        if rnd_num not in rnd_list:
            k+=1
            rnd_list.append(rnd_num)
    return rnd_list


def gaussian(x, mu=0, sigma=1):
    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x-mu)/sigma, 2.)/2.)



if __name__ == "__main__":
    """
    Script schowing how to use curve_fit from scipy.optimize.
    """
    
    from scipy.optimize import curve_fit
    
    def function(x):
        return gaussian(x, 1, 1)

    xdata = np.arange(10)
    ydata = function(xdata) + 0.1*np.random.random(10)

    popt = curve_fit(gaussian, xdata, ydata)

    print(popt[0])
    def fitted_function(x):
        return gaussian(x, popt[0][0], popt[0][1])

    distance = ((function(xdata) - fitted_function(xdata))**2).sum()
    print(distance)