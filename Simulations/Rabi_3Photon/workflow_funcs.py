'''
A set of helper functions for a CPhase simulation. To be used in conjunction with the main.ipynb
in the same directory

Author: Thomas Ersevim, 2026
'''

import sys
sys.path.append('Users/thomasersevim/anaconda3')
sys.path.append('/Users/thomasersevim/QNL/2q_gridium/')

import os
import numpy as np
import scipy
import yaml
import dill
import copy
from scipy.optimize import minimize
from dataclasses import dataclass, asdict
from matplotlib import pyplot as plt
from IPython.display import display, Latex
import qutip as qt

from Circuit_Objs import qchard_evolgates as gates;
from Circuit_Objs.qchard_coupobj import CoupledObjects
from Circuit_Objs.qchard_idealgridium import *
from Circuit_Objs.qchard_fluxonium import *
from Circuit_Objs.qchard_abstractobj import AbstractQubit

__all__ = ['PulseConfig', 'load_qubit', 'state_qubit_state', 'solve',
           'visualize_state_propagation', 'solve_two_photon_drive']

@dataclass
class PulseConfig:
    T_gate: int
    pulse_shape: str
    T_rise: int
    pulse_sigma: float
    DRAG: bool
    DRAG_coeff: float
    drive_amplitude_factor: float
    targeted_drive: list
    drive_type: str # flux, charge
    drive_detuning: float

def load_qubit(qubit:AbstractQubit, dir:str='/Users/thomasersevim/QNL/2q_gridium/etc/qubits/'):
    # Attempt to load the qubits (pre-diagonalized) from a file if it exists

    qubit_path = dir+qubit._save_str()
    if os.path.exists(qubit_path):
        with open(qubit_path, 'rb') as f:
            qubit = dill.load(f)
    return qubit

def state_qubit_state(qubit:AbstractQubit, pulse_cfg1:PulseConfig, pulse_cfg2:PulseConfig):
    print('Pulse 1 targeting transition {} to {}: {:.3f} GHz'.format(
        pulse_cfg1.targeted_drive[0],
        pulse_cfg1.targeted_drive[1],
        qubit.freq(pulse_cfg1.targeted_drive[0], pulse_cfg1.targeted_drive[1])))
    print('Drive detuning: {}'.format(pulse_cfg1.drive_detuning))
    print('Pulse 2 targeting transition {} to {}: {:.3f} GHz'.format(
        pulse_cfg2.targeted_drive[0],
        pulse_cfg2.targeted_drive[1],
        qubit.freq(pulse_cfg2.targeted_drive[0], pulse_cfg2.targeted_drive[1])))
    print('Drive detuning: {}'.format(pulse_cfg2.drive_detuning))
    return

def solve(qubit:AbstractQubit, pulse_cfg1:PulseConfig, pulse_cfg2:PulseConfig, comp_space=[0, 1], solve_method='propagator', mute=False):
    
    # Changes where the charge drive occurs and scales the drive to the size of the matrix element
    drive_type1 = pulse_cfg1.drive_type
    drive_type2 = pulse_cfg2.drive_type
    amp_factor1 = pulse_cfg1.drive_amplitude_factor
    amp_factor2 = pulse_cfg2.drive_amplitude_factor

    if drive_type1=='charge':
        H_drive1 = qubit.n()/np.abs(qubit.n_ij(pulse_cfg1.targeted_drive[0], pulse_cfg1.targeted_drive[1]))
    elif drive_type1=='flux':
        H_drive1 = qubit.phi()/np.abs(qubit.phi_ij(pulse_cfg1.targeted_drive[0], pulse_cfg1.targeted_drive[1]))
    H_drive1 = H_drive1*amp_factor1

    if drive_type2=='charge':
        H_drive2 = qubit.n()/np.abs(qubit.n_ij(pulse_cfg2.targeted_drive[0], pulse_cfg2.targeted_drive[1]))
    elif drive_type2=='flux':
        H_drive2 = qubit.phi()/np.abs(qubit.phi_ij(pulse_cfg2.targeted_drive[0], pulse_cfg2.targeted_drive[1]))
    H_drive2 = H_drive2*amp_factor2

    H_drive = H_drive1 + H_drive2 # BUG: Not a valid way to combine these if they are different types of drives

    omega_d1 = qubit.freq(pulse_cfg1.targeted_drive[0], pulse_cfg1.targeted_drive[1])
    omega_d1 = np.abs(omega_d1)
    omega_d2 = qubit.freq(pulse_cfg2.targeted_drive[0], pulse_cfg2.targeted_drive[1])
    omega_d2 = np.abs(omega_d2)

    t_points1 = np.linspace(0, pulse_cfg1.T_gate, 2 * int(pulse_cfg1.T_gate) + 1)
    t_points2 = np.linspace(0, pulse_cfg2.T_gate, 2 * int(pulse_cfg2.T_gate) + 1)

    p1_dict = asdict(pulse_cfg1)
    p1_dict.update({'t_points': t_points1})
    p1_dict.update({'omega_d': omega_d1})
    p2_dict = asdict(pulse_cfg2)
    p2_dict.update({'t_points': t_points2})
    p2_dict.update({'omega_d': omega_d2})

    max_t_array = t_points1 if len(t_points1) > len(t_points2) else t_points2
    max_T_gate = pulse_cfg1.T_gate if pulse_cfg1.T_gate > pulse_cfg2.T_gate else pulse_cfg2.T_gate

    p_dict_list = [p1_dict, p2_dict]

    if solve_method == 'propagator':
        # This calculates the evolution operator for the whole system  
        U_t = gates.evolution_operator_2phot_microwave(qubit.H(), H_drive, t_points=max_t_array, kwargs=p_dict_list)
    # U_real = gates.change_operator_proj_subspace(qubit, U_t, subspace=comp_space, interaction=interaction)
    # fidelity = gates.fidelity_cz_gate(qubit, U_t, comp_space=comp_space, interaction='off', single_gates='z') # TODO: have another method for all qubits which takes a different number of arguments to account for this interaction argument
    # single_qubit_gates = gates.operator_single_qub_z(system, U_real[-1])
    U_f = U_t[-1]
    U_me = {}
    # for state in comp_space:
        # vec = qubit.eigvec(state)
        # U_me[state] = U_f.matrix_element(vec.dag(), vec)

    if not mute:
        #Note: this is only for unitary evolution. We shall investigate dephasing errors later.
        # print('\nMax fidelity during the simulations: ', np.max(fidelity), 'at', np.argmax(fidelity)/2, 'ns')
        print('** Final values **')
        # print('Fidelity: ', fidelity[-1])
        print('\nDiagonal elements of the evolution operator ' +
            '(amplitudes and phases with respect to E*t in units of pi)')

        initial_state = qubit.eigvec(comp_space[0])
        final_state = qubit.eigvec(comp_space[1])
        # P_driven_transition = gates.prob_transition(U_t, initial_state, final_state)
        # t_2nd_excited = scipy.integrate.trapezoid(P_driven_transition, max_t_array)
        # print('Time spent in the 2nd state for {} - {}: {:.1f} ns'.format(
            # transitions_to_drive[0], transitions_to_drive[1], t_2nd_excited))
        # Like an integrated time so should be proportional to phase accumulation!
    return max_t_array, U_t, 

def visualize_state_propagation(
        qubit:AbstractQubit,
        pulse_cfg1:PulseConfig,
        pulse_cfg2:PulseConfig,
        t_points,
        U_t,
        n_shown_states,
        comp_space=[0,1]):
    # Return graphs with all the relevant details

    # Printing relevant parameters
    plt_00 = {}
    plt_01 = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle('{} with D1 {} driving {} and D2 {} driving {}'.format(
        qubit.name,
        pulse_cfg1.drive_type,
        pulse_cfg1.targeted_drive,
        pulse_cfg2.drive_type,
        pulse_cfg2.targeted_drive,))

    ax000 = axes[0]
    ax001 = axes[1]

    # All plots
    max_final_amp = {}
    eigvecs = qubit.eigvecs()
    psi_t = list(u_i * eigvecs for u_i in U_t)
    for state in range(qubit.nlev): # a-priori we don't know what positions to show, so let's search through all of them, and choose to display the ones with the highest final values (that aren't the identity case)
        plt_00[state] = psi_t[state]
        max_final_amp[state] = np.max(plt_00[state]) # NOTE: Maybe we want to actually show the states with highest intermediate values? (i.e. np.max(plt_00[state])
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax000.scatter(t_points, plt_00[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[0], key))

    max_final_amp = {}
    for state in range(qubit.nlev):
        plt_01[state] = gates.prob_transition(U_t, qubit.eigvec(comp_space[1]), qubit.eigvec(state))
        max_final_amp[state] = plt_01[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax001.scatter(t_points, plt_01[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[1], key))

    # General information
    textfontsize = 18
    fig.text(0.5, 0.16, r'At $t = {}$ ns: '.format(int(t_points[-1])),
            fontsize=textfontsize, ha='center')
    fig.text(0.5, 0.13,
            r'$P({}\rightarrow {}) = {:.4f}$, '.format(
                comp_space[0],
                comp_space[0],
                plt_00[comp_space[0]][-1])
            + r'$P({}\rightarrow {}) = {:.4f}$, '.format(
                comp_space[1],
                comp_space[1],
                plt_01[comp_space[1]][-1]))

    # In plot text for final values
    ax000.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[0],
                comp_space[0], 
                plt_00[comp_space[0]][-1]),
            ha='right', va='top', transform=ax000.transAxes,
            fontsize=textfontsize)
    ax001.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[1],
                comp_space[1],
                plt_01[comp_space[1]][-1]),
            ha='right', va='top', transform=ax001.transAxes,
            fontsize=textfontsize)

    # Below plots text for phase and fidelity
    # fig.text(0.5, 0.1,
    #         r'CZ gate phase accumulation: '
    #         + r'$\phi_{{{}}} + \phi_{{{}}} - \phi_{{{}}} - \phi_{{{}}} = $'.format(
    #             comp_space[0],
    #             comp_space[3],
    #             comp_space[2],
    #             comp_space[1])
    #         + r'${:.3f} \pi $'.format(phase_accum),
    #         fontsize=textfontsize, ha='center');
    fig.text(0.5, 0.05,
            r'Fidelity: '
            + r'$F = {:.6f}$'.format(fidelity[-1]),
            fontsize=textfontsize, ha='center')

    for axarr in axes:
        for ax in axarr:
            ax.legend(loc='lower left')
            ax.set_xlim([np.min(t_points), np.max(t_points)])
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel(r'$P_{i\rightarrow f}$')

    ax000.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[0]))
    ax001.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[1]))

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    
    # TODO: Add meta data to figure
    return fig

def visualize_lost_trace():
    # Graphs trace over time to show losses (one plot is better here)
    return

def minimize_infidelity(system:CoupledObjects, pulse_cfg:PulseConfig, system_cfg:SystemConfig, solve_method='propagator', mute=False, x0=[0,0]):
    def infidelity(x):
        pulse_cfg.drive_detuning, pulse_cfg.DRAG = x
        _, _, _, fidelity = solve(system, pulse_cfg, system_cfg, solve_method, mute=True)
        last_infidelity = 1-fidelity[-1]
        print(last_infidelity)
        return last_infidelity
    
    xopt = minimize(infidelity, x0, method='Nelder-Mead')
    return xopt, infidelity(x=xopt.x)

def converge_on_pi(system:CoupledObjects, pulse_cfg:PulseConfig, system_cfg:SystemConfig, solve_method='propagator', mute=False):
    def cphase_pi_error(x):
        _, _, phase_accum, _ = solve(system, pulse_cfg, system_cfg, solve_method, mute=True)
        print(phase_accum)
        return (np.pi - phase_accum)%(2*np.pi)
    
    xopt = minimize(cphase_pi_error, pulse_cfg.T_gate, method='Nelder-Mead')
    return xopt, cphase_pi_error(x=xopt.x)

def solve_two_photon_drive(qubit, pulse_path1, pulse_path2, n_shown_states=3):
    #combined function that concatenates all the previous functions. Typical entry point

    with open(pulse_path1, 'r') as f:
        pulse_cfg1_dict = yaml.safe_load(f)
        pulse_cfg1 = PulseConfig(**pulse_cfg1_dict)

    with open(pulse_path2, 'r') as f:
        pulse_cfg2_dict = yaml.safe_load(f)
        pulse_cfg2 = PulseConfig(**pulse_cfg2_dict)

    qubit = load_qubit(qubit)
    state_qubit_state(qubit, pulse_cfg1, pulse_cfg1)
    t_points, U_t, phase_accum, fidelity = solve(qubit, pulse_cfg1_dict, pulse_cfg2_dict, solve_method='propagator', mute=False)
    fig = visualize_state_propagation(qubit, pulse_cfg1, pulse_cfg2, t_points, U_t, phase_accum, fidelity, n_shown_states=n_shown_states)
    return fig