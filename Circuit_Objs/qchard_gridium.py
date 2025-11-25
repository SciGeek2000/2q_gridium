# Author: Thomas Ersevim
###########################################################################

"""The Gridium class for representing superconducting gridium qubits.
"""

# NOTE: This gridium is the simple 3 mode gridium toy model.

# Should ultimately have the same number of different gridium objects as there are approximations
# elsewhere (like in the gridium folder)

# For example:
# 3 Mode (Ideal cos(2phi) with ideal phase slip with an inductance and capacitor)
# 4 Mode
# Toy

__all__ = ['Gridium']

import numpy as np
import qutip as qt

class Gridium(object):
    """A class for representing superconducting gridium qubits."""

    def __init__(self, E_L, E_C, E_s, E_2J,
                 phi_ext=np.pi, nlev=5, nlev_lc=50, units='GHz'):
        # Most of these attributes are defined later as properties.
        self.E_L = E_L  # The inductive energy.
        self.E_C = E_C  # The charging energy.
        self.E_s = E_s # The (ideal) phase slip energy
        self.E_2J = E_2J  # The Josephson energy.
        self.phi_ext = phi_ext
        self.nlev = nlev  # The number of eigenstates in the qubit.
        self.nlev_lc = nlev_lc  # The number of states before diagonalization.
        self.units = units
        self.type = 'qubit'

    def __str__(self):
        s = ('A 3-mode gridium qubit with E_L = {} '.format(self.E_L) + self.units
             + ', E_C = {} '.format(self.E_C) + self.units
             + ', E_s = {} '.format(self.E_s) + self.units
             + ', and E_2J = {} '.format(self.E_2J) + self.units
             + '. The external phase shift is phi_ext/pi = {}.'.format(
                    self.phi_ext / np.pi))
        return s

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, value):
        if value <= 0:
            raise Exception('Inductive energy must be positive.')
        self._E_L = value
        self._reset_cache()

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise Exception('Charging energy must be positive.')
        self._E_C = value
        self._reset_cache()

    @property
    def E_s(self):
        return self._E_s

    @E_s.setter
    def E_s(self, value):
        if value <= 0:
            print('*** Warning: Phase slip energy is not positive')
        self._E_s = value
        self._reset_cache()

    @property
    def E_2J(self):
        return self._E_2J

    @E_2J.setter
    def E_2J(self, value):
        if value <= 0:
            print('*** Warning: Josephson energy is not positive. ***')
        self._E_2J = value
        self._reset_cache()

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value):
        self._phi_ext = value
        self._reset_cache()

    @property
    def nlev_lc(self):
        return self._nlev_lc

    @nlev_lc.setter
    def nlev_lc(self, value):
        if value <= 0:
            raise Exception('The number of levels must be positive.')
        self._nlev_lc = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _b_lc(self):
        """Annihilation operator in the LC basis."""
        return qt.destroy(self.nlev_lc)

    def _phi_lc(self):
        """Flux (phase) operator in the LC basis."""
        return (2 * self.E_C / self.E_L) ** (0.25) * qt.position(self.nlev_lc)

    def _n_lc(self):
        """Charge operator in the LC basis."""
        return (self.E_L / (2 * self.E_C)) ** (0.25) * qt.momentum(self.nlev_lc)

    def _hamiltonian_lc(self):
        """Qubit Hamiltonian in the LC basis."""
        E_C = self.E_C
        E_L = self.E_L
        E_s = self.E_s
        E_2J = self.E_2J
        phi = self._phi_lc()
        n = self._n_lc()
        net_phi = phi + self.phi_ext

        cos_2pi_n = 0.5*(np.exp(-2j*np.pi)*n.expm() + np.exp(2j*np.pi)*n.expm())
        cos_2phi = 0.5*(np.exp(-2j)*phi.expm() + np.exp(2j)*phi.expm())
        return E_C * n ** 2 + 0.5 * E_L * net_phi ** 2 - E_s * cos_2pi_n - E_2J * cos_2phi # TODO: Check to make sure coefficients are right here!
        # return E_C * n ** 2 + 0.5 * E_L * net_phi ** 2 - E_s * np.cos(2 * np.pi * n) - E_2J * np.cos(2 * phi) # TODO: Check to make sure coefficients are right here!
        # Will also have to do this as a matrix exponential

    def _eigenspectrum_lc(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates in the LC basis."""
        if not eigvecs_flag:
            if self._eigvals is None:
                H_lc = self._hamiltonian_lc()
                self._eigvals = H_lc.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H_lc = self._hamiltonian_lc()
                self._eigvals, self._eigvecs = H_lc.eigenstates()
            return self._eigvals, self._eigvecs

    def levels(self, nlev=None, eigvecs = False):
        """Eigenenergies of the qubit.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_lc:
            raise Exception('`nlev` is out of bounds.')
        if eigvecs:
            return_tuple = self._eigenspectrum_lc(eigvecs_flag=True)
            return return_tuple[0][:nlev], return_tuple[1][:nlev]
        else:
            return self._eigenspectrum_lc()[:nlev]

    def level(self, level_index, eigvecs=False):
        """Energy of a single level of the qubit.

        Parameters
        ----------
        level_ind : int
            The qubit level starting from zero.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_index < 0 or level_index >= self.nlev_lc:
            raise Exception('The level is out of bounds')
        if eigvecs:
            return_tuple = self.levels(eigvecs = True)
            return return_tuple[0][level_index], return_tuple[1][level_index]
        else:
            return self.levels()[level_index]

    def eigvec(self, level_index):
        """A shortcut to get an eigenvector via level(eigvec=True).

        Returns
        -------
        :class:`qutip.Qobj`
            Eigenvector.
        """
        _, evec = self.level(level_index=level_index, eigvecs=True)
        return evec

    def freq(self, level1, level2):
        """Transition energy/frequency between two levels of the qubit.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2` defined
            as the difference of energies. Positive if `level1` < `level2`.
        """
        return self.level(level2) - self.level(level1)

    def H(self, nlev=None):
        """Qubit Hamiltonian in its eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The Hamiltonian operator.
        """
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    def eye(self, nlev=None):
        """Identity operator in the qubit eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_lc:
            raise Exception('`nlev` is out of bounds.')
        return qt.qeye(nlev)

    def phi(self, nlev=None):
        """Generalized-flux operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The flux operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_lc:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        phi_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                phi_op[ind1, ind2] = self._phi_lc().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(phi_op)

    def n(self, nlev=None):
        """Charge operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The charge operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev_lc:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        n_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                n_op[ind1, ind2] = self._n_lc().matrix_element(
                    evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(n_op)

    def phi_ij(self, level1, level2):
        """The flux matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        complex
            The matrix element of the flux operator.
        """
        if (level1 < 0 or level1 > self.nlev_lc
                or level2 < 0 or level2 > self.nlev_lc):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._phi_lc().matrix_element(
            evecs[level1].dag(), evecs[level2])

    def n_ij(self, level1, level2):
        """The charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        complex
            The matrix element of the charge operator.
        """
        if (level1 < 0 or level1 > self.nlev_lc
                or level2 < 0 or level2 > self.nlev_lc):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_lc(eigvecs_flag=True)
        return self._n_lc().matrix_element(evecs[level1].dag(), evecs[level2])