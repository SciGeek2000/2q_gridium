# Author: Thomas Ersevim, 2026
###################################################################################################

'''A Transmon (defined using LC basis) test suite'''

import pytest
import scqubits #https://scqubits.readthedocs.io/en/latest/installation.html#arm64-compatibility
import sys

sys.path.append('/Users/thomasersevim/QNL/2q_gridium')
import Circuit_Objs.qchard_transmon as transmon

def test_eigenvals():
    # Ensures the eigenvalues of the qchard transmon are within 2% of known scqubits case!
    scTransmon = scqubits.Transmon(EC=1, EJ=50, ng=0, ncut=15)
    zeroed_scTransmon_vals = (scTransmon.eigenvals() - scTransmon.eigenvals()[0])[1:]
    
    qchard_transmon = transmon.Transmon(E_C=1, E_J=50, n_g=0, nlev=6, nlev_lc=25, units='GHz')
    zeroed_qchardTransmon_vals = qchard_transmon.transition_energies()[1:]
    
    remainder = (1 - (zeroed_qchardTransmon_vals/zeroed_scTransmon_vals))
    max_off = max(abs(remainder))
    assert max_off < 0.02
