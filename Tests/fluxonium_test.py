# Author: Thomas Ersevim, 2026
###################################################################################################

'''An Fluxonium test suite'''

import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

sys.path.append('/Users/thomasersevim/QNL/2q_gridium')
from Circuit_Objs.qchard_fluxonium import *
import Notebooks.plotting_settings

fluxonium = Fluxonium(**fluxonium_params, **std_fluxonium_sim_params)

def sweep_param_spectrum(sweep_array:np.ndarray, param_name:str, tested_fluxonium:Fluxonium) -> np.ndarray:
    ''' Returns an array of energies based on a changing parameter "parameter name" using sweep_array as the data array.'''
    copied_gridium = copy.deepcopy(tested_fluxonium)
    spectrum_array = np.empty((*np.shape(sweep_array), tested_fluxonium.nlev))
    for i, value in enumerate(sweep_array):
        print(i)
        setattr(copied_gridium, param_name, value)
        spectrum_array[i] = copied_gridium.transition_energies()
    return spectrum_array

## Tests ##
# All tests currently agree well with Dat's simulations (when given a sufficently high lc cutoff)

@pytest.mark.parametrize('tested_fluxonium', [fluxonium]) # TODO: Change this so that it is really checking symmetry around phi_ext=pi (doesn't have pi periodicitiy)
def test_phi_ext_spectrum(tested_fluxonium, plotting=False):
    '''Tests phi external spectrum is pi periodic and/or plots the spectrum'''

    error_threshold = 1e-4
    checking_points = 4

    if plotting == False: 
        fluxes = np.linspace(0, np.pi, checking_points+1)[1:]
        right_spectrum = sweep_param_spectrum(np.pi + fluxes, 'phi_ext', tested_fluxonium)
        left_spectrum = sweep_param_spectrum((np.pi - fluxes), 'phi_ext', tested_fluxonium)
        max_pct_errors = list()
        for lev in range(tested_fluxonium.nlev-1): # Removing all zero transition
            pct_error = (right_spectrum[:,lev+1]-left_spectrum[:,lev+1])/np.max(np.abs(right_spectrum[:,lev+1]))
            max_pct_error = np.max(np.abs(pct_error))
            max_pct_errors.append(max_pct_error)
        worst_error = np.max(np.abs(max_pct_errors)) 
        print('List of max percent errors: {}\n'.format(max_pct_errors))
        print('Worst percent error: {}'.format(worst_error))
        assert worst_error < error_threshold # Assertion is that phi_ext values + pi should be identical (within threshold)

    if plotting == True: # To visually ensure that the convergence is good and stable.
        fluxes = np.linspace(0, 2*np.pi, 31)
        right_spectrum = sweep_param_spectrum(fluxes, 'phi_ext', tested_fluxonium)
        right_spectrum = right_spectrum[:, 1:]
        for i in range(right_spectrum.shape[1]):
            plt.plot(fluxes/(2*np.pi), right_spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel(r'$\phi_{ext}/2\pi$')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_fluxonium.units))
        plt.legend()
        plt.show()
        return

@pytest.mark.parametrize('tested_fluxonium', [fluxonium]) # TODO: could turn this into an actual test. perhaps like that degeneracies are degeneracies or something.
def test_ng_spectrum(tested_fluxonium):
    '''Tests ng dispersion for agreement with Dat's previous spectrum results'''
    ngs = np.linspace(0, 1, 41) #Is symmetric about 0 with periodicity of 1, so only doing this range
    spectrum = sweep_param_spectrum(ngs, 'ng', tested_fluxonium)
    zeros = tested_fluxonium.transition_energies()
    zeroed_spectrum = (spectrum-zeros)[:,1:]
    for i in range(zeroed_spectrum.shape[1]):
        plt.plot(ngs, zeroed_spectrum[:, i], label=r'0$\rightarrow$' + str(i+1))
    plt.xlabel(r'$n_g$')
    plt.ylabel(r'Transition Energy ({}/h)'.format(tested_fluxonium.units))
    plt.legend()
    plt.tight_layout()
    plt.show()

@pytest.mark.parametrize('tested_fluxonium', [fluxonium])
def test_convergence(tested_fluxonium, plotting=False):
    '''
    A test for the convergence of the gridium spectra so that as a function of the nlev_lc, the
    change is negligible (i.e. the asymptote gives within less than 1 percent of the chosen
    nlev_lc cutoff value.
    
    :param tested_gridium: The gridium used for the convergence test
    '''

    copied_fluxonium = copy.deepcopy(tested_fluxonium)
    error_threshold = 1e-4
    nlev = copied_fluxonium.nlev

    if plotting == False: 
        default_lc_transitions = copied_fluxonium.transition_energies(nlev=nlev)[1:]
        copied_fluxonium.nlev_lc = copied_fluxonium.nlev_lc*2
        twice_lc_transitions = copied_fluxonium.transition_energies(nlev=nlev)[1:]
        pct_error = (default_lc_transitions-twice_lc_transitions)/np.max(np.abs(twice_lc_transitions))
        worst_error = np.max(np.abs(pct_error)) 
        assert worst_error < error_threshold # For testing, the assertion is that the precent error, scaled to the largest transition (to get around zeros) should be < error_threshold

    if plotting == True: # To visually ensure that the convergence is good and stable.
        plotting_points = 20
        nlevs_lc = np.linspace(10,copied_fluxonium.nlev_lc*2, plotting_points, dtype=int)
        convergence_array = np.empty((plotting_points, nlev))
        for i, nlev_lc in enumerate(nlevs_lc):
            copied_fluxonium.nlev_lc = nlev_lc
            convergence_array[i,:] = copied_fluxonium.transition_energies(nlev=nlev)
        for i in range(nlev):
            plt.plot(nlevs_lc, convergence_array[:,i], label=r'0$\rightarrow$' + str(i+1))
        plt.xlabel('H.O. Basis Cutoff Dimentionality')
        plt.ylabel(r'Transition Energy ({}/h)'.format(tested_fluxonium.units))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

# TODO: Implement eigenvector plots -- will help with understanding.


'''
phi_matrix_elements = np.conj(states.T)@system['phi_ope']@states
n_matrix_elements = np.conj(states.T)@system['n_ope']@states

plt.figure(figsize = [2,2])
level_max=4
matrix = np.abs(phi_matrix_elements[0:level_max,0:level_max])
plt.rcParams.update({'font.size': 6})
# Plot the matrix cmap = 'viridis'
plt.imshow(matrix)
# Add text annotations for each cell
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white')
plt.yticks(range(matrix.shape[0]))
plt.colorbar(shrink=0.8)  # Add a color bar
plt.title(r'$| \langle j  | \hat \phi | k \rangle |$')
plt.xlabel('eigenstates')
plt.ylabel('eigenstates')
plt.show()
'''

test_phi_ext_spectrum(fluxonium, plotting=False)
# test_ng_spectrum(fluxonium)
# test_convergence(plotting=False, tested_fluxonium=fluxonium)