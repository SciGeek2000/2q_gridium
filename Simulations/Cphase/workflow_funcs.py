# from Simulations.Cphase.fluxonium_idealgridium_setup import *;

import sys
sys.path.append('Users/thomasersevim/anaconda3')
sys.path.append('/Users/thomasersevim/QNL/2q_gridium/')

import os
import numpy as np
import scipy
import yaml
import dill
from scipy.optimize import minimize
from dataclasses import dataclass
from matplotlib import pyplot as plt
from IPython.display import display, Latex


from Circuit_Objs import qchard_evolgates as gates;
from Circuit_Objs.qchard_coupobj import CoupledObjects
from Circuit_Objs.qchard_idealgridium import *
from Circuit_Objs.qchard_fluxonium import *

__all__ = ['PulseConfig', 'SystemConfig', 'load_qubits', 'couple_qubits', 'solve',
           'visualize_state_propagation', 'solve_coupled_qubits', 'minimize_infidelity', 'converge_on_pi']

@dataclass
class PulseConfig:
    T_gate: int
    pulse_shape: str
    pulse_sigma: float
    DRAG: bool
    DRAG_coeff: float
    drive_amplitude_factor: float
    drive_detuning: float

@dataclass
class SystemConfig:
    qubitA: str
    qubitB: str
    driven_qubit: str
    J_C: float
    interaction: str
    coupling_type:str
    computational_space: list
    transitions_to_drive: list
    detuned_transitions: list
    comparitive_transitions: list
    detuned_comparitive_transitions: list
    coupled_resonant_transitions: list

def load_qubits(qubitA, qubitB, dir:str='/Users/thomasersevim/QNL/2q_gridium/etc/qubits/'):
    # Attempt to load the qubits (pre-diagonalized) from a file if it exists

    qubitA_path = dir+qubitA._save_str()
    if os.path.exists(qubitA_path):
        with open(qubitA_path, 'rb') as f:
            qubitA = dill.load(f)
    
    qubitB_path = dir+qubitB._save_str()
    if os.path.exists(qubitB_path):
        with open(qubitB_path, 'rb') as f:
            qubitB = dill.load(f)

    return qubitA, qubitB

def couple_qubits(qubitA, qubitB, system_cfg:SystemConfig, pulse_cfg:PulseConfig, mute=False):
    # Couple qubits and define driven qubit(s). Report back relevant information

    transitions_to_drive = system_cfg.transitions_to_drive
    comparitive_transitions = system_cfg.comparitive_transitions
    detuned_transitions =  system_cfg.detuned_transitions
    detuned_comparitive_transitions = system_cfg.detuned_comparitive_transitions
    coupling_type = system_cfg.coupling_type
    J_C = system_cfg.J_C

    drive_detuning = pulse_cfg.drive_detuning
    drive_amplitude_factor = pulse_cfg.drive_amplitude_factor

    system = CoupledObjects(qubitA, qubitB, [qubitA, qubitB, J_C, coupling_type])

    # Calculate the drive frequency.
    omega_d = abs(system.freq(transitions_to_drive[0], transitions_to_drive[1])) + drive_detuning
    driven_transition_detuning = (
        system.freq(transitions_to_drive[0], transitions_to_drive[1]) -
        system.freq(detuned_transitions[0], detuned_transitions[1]))
    comparitive_transition_detuning = (
        system.freq(comparitive_transitions[0], comparitive_transitions[1]) - 
        system.freq(detuned_comparitive_transitions[0], detuned_comparitive_transitions[1]))
    relative_detuning = driven_transition_detuning - comparitive_transition_detuning

    if not mute:
        print('Detuning between {}-{} and {}-{}: {:.1f} MHz'.format(
            transitions_to_drive[0], transitions_to_drive[1],
            detuned_transitions[0], detuned_transitions[1],
            1000*np.abs(driven_transition_detuning)))
        print('Detuning between {}-{} and {}-{}: {:.1f} MHz'.format(
            comparitive_transitions[0], comparitive_transitions[1],
            detuned_comparitive_transitions[0], detuned_comparitive_transitions[1],
            1000*np.abs(comparitive_transition_detuning)))
        print('Relative detuning is {:.3f} MHz'.format(1000*np.abs(relative_detuning)))
        print('Transition to drive: {} - {} with frequency {:.4f} GHz'.format(
            transitions_to_drive[0], transitions_to_drive[1],
            abs(system.freq(transitions_to_drive[0], transitions_to_drive[1]))))
        print('Drive frequency: {:.4f} GHz'.format(omega_d))
        print('Drive amplitude scale factor: {:.4f}'.format( drive_amplitude_factor))
    return system

def solve(system:CoupledObjects, pulse_cfg:PulseConfig, system_cfg:SystemConfig, solve_method='propagator', mute=False):
    # Solve according to parts

    driven_qubit = system_cfg.driven_qubit
    level1 = system_cfg.transitions_to_drive[0]
    level2 = system_cfg.transitions_to_drive[1]
    comp_space = system_cfg.computational_space
    interaction = system_cfg.interaction
    
    drive_amplitude_factor = pulse_cfg.drive_amplitude_factor
    T_gate = pulse_cfg.T_gate
    shape = pulse_cfg.pulse_shape
    sigma = pulse_cfg.pulse_sigma
    drag = pulse_cfg.DRAG
    drag_coeff = pulse_cfg.DRAG_coeff
    
    omega_d = abs(system.freq(level1, level2)) + pulse_cfg.drive_detuning #TODO: not clean

    qubitA = system._objects[0]
    qubitB = system._objects[1]

    if driven_qubit=='A': # TODO: This gives strange results for SCQTransmons
        H_drive = system.n(0)/np.abs(system.n_ij(qubitA, level1, level2, interaction='on'))*drive_amplitude_factor
        # H_drive_dummy = 0 * system.n(0)
    elif driven_qubit=='B':
        H_drive = system.n(1)/np.abs(system.n_ij(qubitB, level1, level2, interaction='on'))*drive_amplitude_factor
        # H_drive_dummy = 0 * system.n(1)
    elif driven_qubit=='AB': # TODO: check this case regarding how the n_ij matrices should be combined. Seems like an identical drive strength on each for this.
        H_drive = (system.n(0) + system.n(1))/np.abs(
            (system.n_ij(qubitA, level1, level2, interaction='on')*drive_amplitude_factor +
            system.n_ij(qubitB, level1, level2, interaction='on')*drive_amplitude_factor))
        # H_drive_dummy = 0 * system.n(0) + system.n(1)

    t_points = np.linspace(0, T_gate, 2 * int(T_gate) + 1)
    if solve_method == 'sesolve':
        # This calculates the evolution operator that works for
        # computational levels only.
        U_t = gates.evolution_compspace_microwave(system.H(), H_drive, comp_space=comp_space, t_points=t_points,T_gate=T_gate, shape=shape, sigma=sigma, omega_d=omega_d,interaction=interaction)
    elif solve_method == 'propagator':
        # This calculates the evolution operator for the whole system  
        U_t = gates.evolution_operator_microwave(system.H(), H_drive, comp_space=comp_space, t_points=t_points,T_gate=T_gate, shape=shape, sigma=sigma, DRAG = drag, DRAG_coefficient = drag_coeff, omega_d=omega_d,interaction=interaction);
        # U_nothing = gates.evolution_operator_microwave(-system.H(), H_drive_dummy, comp_space=comp_space, t_points=t_points,T_gate=T_gate, shape=shape, sigma=sigma, DRAG = drag, DRAG_coefficient = drag_coeff, omega_d=omega_d,interaction=interaction);

    U_real = gates.change_operator_proj_subspace(system, U_t, subspace=comp_space, interaction=interaction)
    single_qubit_gates = gates.operator_single_qub_z(system, U_real[-1])
    fidelity = gates.fidelity_cz_gate(system, U_t, comp_space=comp_space, interaction=interaction, single_gates='z')
    U_f = U_t[-1]
    U_me = {}
    for state in comp_space:
        vec = system.eigvec(state, interaction=interaction)
        U_me[state] = U_f.matrix_element(vec.dag(), vec)
    phase_accum = (np.angle(U_me[comp_space[0]]) + np.angle(U_me[comp_space[3]]) -
                   np.angle(U_me[comp_space[1]]) - np.angle(U_me[comp_space[2]]))
    phase_accum = phase_accum / np.pi

    if not mute:
        #Note: this is only for unitary evolution. We shall investigate dephasing errors later.
        print('max fidelity during the simulations: ', np.max(fidelity))
        print('\n** Final values **')
        print('Fidelity: ', fidelity[-1])
        print('Diagonal elements of the evolution operator ' +
            '(amplitudes and phases with respect to E*t in units of pi)')
        for state in comp_space:
            print(state, np.abs(U_me[state]),
                (np.angle(U_me[state] * np.exp(2j * np.pi * system.level(state) * T_gate))) / np.pi)
        display(Latex(r'$(\phi_{{{}}} + \phi_{{{}}} - \phi_{{{}}} - \phi_{{{}}})/\pi=$ {}'.format(
            comp_space[0], comp_space[3], comp_space[1], comp_space[2], phase_accum)))

        initial_state = system.eigvec(level1)
        final_state = system.eigvec(level2)
        P_driven_transition = gates.prob_transition(U_t, initial_state, final_state)
        t_2nd_excited = scipy.integrate.trapezoid(P_driven_transition, t_points)
        print('Time spent in the 2nd state for {} - {}: {:.1f} ns'.format(
            level1, level2, t_2nd_excited))
        # Like an integrated time so should be proportional to phase accumulation!

    return t_points, U_t, phase_accum, fidelity

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

# Want a pulse length optimizer for pi rotation then minimize infidelity!
#Gate parameter
'''
T_gate_array = np.linspace(20,100,41) #ns
error_array = np.zeros_like(T_gate_array)
drag_coeff_array = np.zeros_like(T_gate_array)
delta_omega_d_array = np.zeros_like(T_gate_array)

for T_idx, T_gate in enumerate(T_gate_array):
  
    t_points = np.linspace(0, T_gate, 10 * int(T_gate) + 1)
    delta_omega_d = 0
    drag_coeff = 0

    x0 = [delta_omega_d, drag_coeff]
    xopt = minimize(infidelity, x0, method ='Powell', tol = 1e-6)
    
    error_array[T_idx] = infidelity(xopt.x)
    delta_omega_d[T_idx] = xopt.x[0]
    drag_coeff[T_idx] = xopt.x[1]
'''

def visualize_state_propagation(
        system:CoupledObjects,
        system_cfg:SystemConfig,
        t_points,
        U_t,
        phase_accum,
        fidelity,
        n_shown_states):
    # Return graphs with all the relevant details

    qubitA = system._objects[0]
    qubitB = system._objects[1]
    comp_space = system_cfg.computational_space

    # Printing relevant parameters
    plt_00 = {}
    plt_01 = {}
    plt_10 = {}
    plt_11 = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ax000 = axes[0, 0]
    ax001 = axes[0, 1]
    ax010 = axes[1, 0]
    ax011 = axes[1, 1]

    all_states_str = []
    for i in range(qubitA.nlev):
        for j in range(qubitB.nlev):
            all_states_str.append(f"{i}{j}")

    # All plots
    max_final_amp = {}
    for state in all_states_str: # a-priori we don't know what positions to show, so let's search through all of them, and choose to display the ones with the highest final values (that aren't the identity case)
        plt_00[state] = gates.prob_transition(U_t, system.eigvec(comp_space[0]), system.eigvec(state))
        max_final_amp[state] = plt_00[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax000.scatter(t_points, plt_00[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[0], key))

    max_final_amp = {}
    for state in all_states_str:
        plt_01[state] = gates.prob_transition(U_t, system.eigvec(comp_space[1]), system.eigvec(state))
        max_final_amp[state] = plt_01[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax001.scatter(t_points, plt_01[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[1], key))

    max_final_amp = {}
    for state in all_states_str:
        plt_10[state] = gates.prob_transition(U_t, system.eigvec(comp_space[2]), system.eigvec(state))
        max_final_amp[state] = plt_10[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax010.scatter(t_points, plt_10[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[2], key))

    max_final_amp = {}
    for state in all_states_str:
        plt_11[state] = gates.prob_transition(U_t, system.eigvec(comp_space[3]), system.eigvec(state))
        max_final_amp[state] = plt_11[state][-1]
    top_keys = sorted(max_final_amp, key=max_final_amp.get, reverse=True)[:n_shown_states]
    for key in top_keys:
        ax011.scatter(t_points, plt_11[key], lw=2, label=r'$P({}\rightarrow{})$'.format(comp_space[3], key))

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
                plt_01[comp_space[1]][-1])
            + r'$P({}\rightarrow {}) = {:.4f}$, '.format(
                comp_space[2],
                comp_space[2],
                plt_10[comp_space[2]][-1])
            + r'$P({}\rightarrow {}) = {:.4f}$'.format(
                comp_space[3],
                comp_space[3],
                plt_11[comp_space[3]][-1]),
            fontsize=textfontsize, ha='center')

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
    ax010.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[2],
                comp_space[2],
                plt_10[comp_space[2]][-1]),
            ha='right', va='top', transform=ax010.transAxes,
            fontsize=textfontsize)
    ax011.text(0.98, 0.93,
            r'$P({} \rightarrow {}) = {:.6f}$'.format(
                comp_space[3],
                comp_space[3],
                plt_11[comp_space[3]][-1]),
            ha='right', va='top', transform=ax011.transAxes,
            fontsize=textfontsize)

    # Below plots text for phase and fidelity
    fig.text(0.5, 0.1,
            r'CZ gate phase accumulation: '
            + r'$\phi_{{{}}} + \phi_{{{}}} - \phi_{{{}}} - \phi_{{{}}} = $'.format(
                comp_space[0],
                comp_space[3],
                comp_space[2],
                comp_space[1])
            + r'${:.3f} \pi $'.format(phase_accum),
            fontsize=textfontsize, ha='center');
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
    ax010.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[2]))
    ax011.set_title(
        r'Starting in $|{}\rangle$'.format(comp_space[3]))

    fig.tight_layout(rect=[0, 0.15, 1, 1]);
    return fig

def visualize_lost_trace():
    # Graphs trace over time to show losses (one plot is better here)
    return

def solve_coupled_qubits(qubitA, qubitB, pulse_path:str, syscfg_path:str, n_shown_states=3):
    #combined function that concatenates all the previous functions. Typical entry point

    with open(pulse_path, 'r') as f:
        data = yaml.safe_load(f)
        pulse_cfg = PulseConfig(**data)

    with open(syscfg_path, 'r') as f:
        data = yaml.safe_load(f)
        system_cfg = SystemConfig(**data)
        del data
 
    qubitA, qubitB = load_qubits(qubitA, qubitB)
    system = couple_qubits(qubitA, qubitB, system_cfg, pulse_cfg, mute=False)
    t_points, U_t, phase_accum, fidelity = solve(system, pulse_cfg, system_cfg, solve_method='propagator', mute=False)
    fig = visualize_state_propagation(system, system_cfg, t_points, U_t, phase_accum, fidelity, n_shown_states=n_shown_states)
    # visualize_lost_trace()
    return fig


# Given transition of qubitA, return qubitB scaled so that is the transition energy level (simple scaling)