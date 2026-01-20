# Author: Thomas Ersevim, 2026
###################################################################################################

'''An IdealGridium test suite'''

import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/thomasersevim/QNL/2q_gridium')
from Circuit_Objs.qchard_idealgridium import *

soft_IdealGridium = IdealGridium(*soft_IdealGridium_params,
                                 *std_IdealGridium_sim_params)
hard_IdealGridium = IdealGridium(*hard_IdealGridium_params,
                                 *std_IdealGridium_sim_params)

def test_spectrum():
    pass

def test_eigenvals():
    pass


"""
def test_eigenvals():
    qchard_gridium = IdealGridium(**hard_gridium_params,
        ng=0,
        phi_ext=np.pi,
        nlev=7,
        nlev_lc=100,
        units='GHz',
    ) 
    try:
        _qchard_evals = qchard_gridium.levels()
    finally:
        pass
    qchard_evals = (_qchard_evals-_qchard_evals[0])
    # assert np.all(qchard_evals, [0,0]) # TODO: Add known output
    return qchard_evals


def gridium_spectrum(fluxes: np.ndarray, gridium:IdealGridium) -> np.ndarray:
    '''Returns an array of energies based on input flux'''
    spectrum_array = np.empty((*np.shape(fluxes), gridium.nlev))
    for i, flux in enumerate(fluxes):
        gridium.phi_ext = flux
        _evals = gridium.levels()
        evals = (_evals - _evals[0])
        spectrum_array[i] = evals
    return spectrum_array

if __name__=='__main__':
    # print(test_eigenvals())
    evals = gridium_spectrum(np.linspace(0, 2*np.pi, 100), IdealGridium(**soft_gridium_params, ng=0, phi_ext=np.pi, nlev=7, nlev_lc=100, units='GHz',))
    print(evals)
    plt.plot(evals)
    plt.show()

"""

# TODO: Implement a graphing scheme for the eigenstates for visual confirmation