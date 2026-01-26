# Author: Thomas Ersevim, 2026
###################################################################################################

'''An IdealGridium test suite'''

import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

sys.path.append('/Users/thomasersevim/QNL/2q_gridium')
from Circuit_Objs.qchard_idealgridium import *
import Notebooks.plotting_settings

soft_IdealGridium = IdealGridium(**soft_IdealGridium_params,
                                 **std_IdealGridium_sim_params)
hard_IdealGridium = IdealGridium(**hard_IdealGridium_params,
                                 **std_IdealGridium_sim_params)
hard_IdealGridium.nlev_lc = 600

def sweep_param_spectrum(sweep_array:np.ndarray, param_name:str, gridium:IdealGridium) -> np.ndarray:
    ''' Returns an array of energies based on a changing parameter "parameter name" using sweep_array as the data array.'''
    copied_gridium = copy.deepcopy(gridium)
    spectrum_array = np.empty((*np.shape(sweep_array), gridium.nlev))
    for i, value in enumerate(sweep_array):
        print(value)
        setattr(copied_gridium, param_name, value)
        spectrum_array[i] = copied_gridium.transition_energies()
    return spectrum_array

## Tests ##
# All tests currently agree well with Dat's simulations (when given a sufficently high lc cutoff)

@pytest.mark.parameterize('tested_gridium', [soft_IdealGridium, hard_IdealGridium])
def test_phi_ext_spectrum(tested_gridium=soft_IdealGridium):
    '''Tests phi external spectrum for agreement with Dat's previous spectrum results'''
    fluxes = np.linspace(0, 2*np.pi, 3)
    spectrum = sweep_param_spectrum(fluxes, 'phi_ext', tested_gridium)
    spectrum = spectrum[:, 1:]
    for i in range(spectrum.shape[1]):
        plt.plot(fluxes, spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
    plt.xlabel(r'$\phi_{ext}/2\pi$')
    plt.ylabel(r'Transition Energy ({}/h)'.format(tested_gridium.units))
    plt.legend()
    plt.show()

@pytest.mark.parameterize('tested_gridium', [soft_IdealGridium, hard_IdealGridium])
def test_ng_spectrum(tested_gridium=soft_IdealGridium):
    '''Tests ng dispersion for agreement with Dat's previous spectrum results'''
    ngs = np.linspace(0, 1, 41) #Is symmetric about 0 with periodicity of 1, so only doing this range
    spectrum = sweep_param_spectrum(ngs, 'ng', tested_gridium)
    zeros = tested_gridium.transition_energies()
    zeroed_spectrum = (spectrum-zeros)[:,1:]
    for i in range(zeroed_spectrum.shape[1]):
        plt.plot(ngs, zeroed_spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
    plt.xlabel(r'$n_g$')
    plt.ylabel(r'Transition Energy ({}/h)'.format(tested_gridium.units))
    plt.legend()
    plt.show()


# TODO: Implement eigenvector plots -- will help with understanding.
# TODO: Saving figures with version metadata
# TODO: Move most of this stuff to the object itself for later use.

# test_phi_ext_spectrum()
# test_ng_spectrum()

print(soft_IdealGridium.transition_energies())

# Example parameterized test! https://docs.pytest.org/en/stable/how-to/parametrize.html#parametrize
#@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
# def my_test(test_input, expected):
    # assert 1