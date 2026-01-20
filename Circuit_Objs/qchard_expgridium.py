# Author: Thomas Ersevim, 2026
#########################################################################

'''Establishes the ExpGridium class for modeling the gridium qubit and for its use in qchard.'''

__all__ = ['ExpGridium', 'soft_ExpGridium_params', 'hard_ExpGridium_params', 'std_ExpGridium_operation_params', 'std_ExpGridium_sim_params']

import numpy as np
import qutip as qt

# Defines intrinsic circuit parameters for later ease of use
soft_ExpGridium_params = { # Like regime 'a' in Table S1 in https://arxiv.org/pdf/2509.14656
    'E_C': 0.5,
    'E_J': 5,
    'E_Lk': 1,
    'E_L': 1,
    'E_Js': 4,
    'E_Cs': 8,
}
hard_ExpGridium_params = { # Like regime 'd' in Table S1 in https://arxiv.org/pdf/2509.14656
    'E_C': 0.5,
    'E_J': 10,
    'E_Lk': 0.2,
    'E_L': 0.2,
    'E_Js': 4,
    'E_Cs': 8,
}
# Defines standard environmental operating conditions of the ExpGridium
std_ExpGridium_operation_params = {
    'ng': 0,
    'phi_ext': 0,
    'theta_ext': np.pi,
}
# Defines the standard simulation conditions of the ExpGridium
std_ExpGridium_sim_params = {
    'nlev': 6,
    'nlev_lc': 99,
    'units': 'GHz',
}

class ExpGridium(object):
    '''
    A class for representing the experimentally achievable, floating, gridium qubit

    Parameter regieme of interest: E_Cs > E_Js >> E_L and E_J >> E_C, E_Lk
    
    This is a fully detailed experimetal model, and is therefore the most expensive to run

    Should be a three mode model.
    '''

    def __init__(self, E_C, E_J, E_Lk, E_L, E_Js, E_Cs,
                 ng=0, phi_ext=0, theta_ext=np.pi,
                 nlev=6, nlev_lc=99, units='GHz'):
        self.E_C = E_C, # The globally confining capacitive energy (patterned pads in real circuit)
        self.E_J = E_J, # The individual EJ of each junction in the KITE (no asymmetry).
        self.E_Lk = E_Lk, # The inductance in each leg of the KITE
        self.E_L = E_L, # The globally confining inductance 
        self.E_Js = E_Js, # The josephson junction acting as a phase slip element
        self.E_Cs = E_Cs, # The capacitance inherent to the phase slip josephson junction
        self.ng = ng, # The offset gate charge
        self.phi_ext = phi_ext, # The threaded flux (not through the KITE)
        self.theta_ext = theta_ext # The threaded flux through the KITE
        self.nlev = nlev, # The number of eigenstates to solve in the qubit
        self.nlev_lc = nlev_lc, # The number of coherent states to use in the diagonalization
        self.units = units,
        self.type = 'qubit'
        # Could add any number of extra paracitic capacitance terms if so inclined

    def __str__(self):
        s = ('A multi-mode gridium qubit with E_C = {} '.format(self.E_C) + self.units
            + ', E_J = {} '.format(self.E_J) + self.units
            + ', E_Lk = {} '.format(self.E_Lk) + self.units
            + ', E_L = {} '.format(self.E_L) + self.units
            + ', E_Js = {} '.format(self.E_Js) + self.units
            + ', E_Cs = {} '.format(self.E_Cs) + self.units
            + '. The external phase shift is phi_ext/pi = {}. '.format(self.phi_ext/np.pi)
            + 'The phase shift through the KITE is theta_ext/pi = {}. '.format(self.theta_ext/np.pi)
        )
        return s

