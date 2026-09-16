"""IdealGridium flux-drive helpers for the Rabi_3Photon workflow.

This module only supplies drive plumbing. It does not select a three-photon
pathway, calibrate a flux line, or optimize pulse parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import dill
from matplotlib import pyplot as plt
import numpy as np
import qutip as qt
import yaml

from Circuit_Objs import qchard_evolgates as gates
from Circuit_Objs.qchard_idealgridium import IdealGridium

__all__ = [
    'PulseConfig',
    'load_qubit',
    'load_pulse_config',
    'logical_gate_target',
    'evaluate_logical_gate',
    'state_qubit_state',
    'solve',
    'solve_multitone_drive',
    'solve_two_photon_drive',
    'solve_three_tone_drive',
    'visualize_state_propagation',
]


@dataclass
class PulseConfig:
    """Configuration for one independently timed flux-drive tone.

    Frequencies and detunings are in GHz, and times are in ns. If
    ``drive_frequency`` is omitted, the carrier is the absolute targeted
    IdealGridium transition frequency plus ``drive_detuning``. If it is
    supplied, the detuning is added to that explicit carrier instead.
    """

    T_gate: float
    pulse_shape: str
    targeted_drive: Sequence[int]
    drive_amplitude_factor: float = 1.0
    drive_detuning: float = 0.0
    carrier_phase: float = 0.0
    T_start: float = 0.0
    T_rise: float | None = None
    pulse_sigma: float = 0.25
    DRAG: bool = False
    DRAG_coeff: float = 0.0
    drive_frequency: float | None = None
    drive_type: str = 'flux'

    def __post_init__(self):
        # The original YAML files use the scalar ``None``, which PyYAML reads
        # as a string. Accept it as the intended absent rise time.
        if self.T_rise == 'None':
            self.T_rise = None
        if len(self.targeted_drive) != 2:
            raise ValueError('targeted_drive must contain exactly two levels.')
        if self.T_gate <= 0:
            raise ValueError('T_gate must be positive.')
        if self.T_start < 0:
            raise ValueError('T_start must be non-negative.')


def load_qubit(qubit: IdealGridium, directory: str | Path | None = None):
    """Load a cached IdealGridium when an explicit cache directory is given."""
    _require_idealgridium(qubit)
    if directory is None:
        return qubit

    qubit_path = Path(directory) / qubit._save_str()
    if qubit_path.exists():
        with qubit_path.open('rb') as stream:
            qubit = dill.load(stream)
        _require_idealgridium(qubit)
    return qubit


def load_pulse_config(path: str | Path) -> PulseConfig:
    """Load one tone without silently changing its drive type or transition."""
    with Path(path).open('r') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError('Pulse configuration must be a YAML mapping.')
    return PulseConfig(**data)


def _require_idealgridium(qubit):
    if not isinstance(qubit, IdealGridium):
        raise TypeError('Rabi_3Photon currently supports IdealGridium only.')


def _coerce_pulse_configs(
        pulse_cfg1: PulseConfig | Sequence[PulseConfig],
        pulse_cfg2: PulseConfig | None = None,
        pulse_cfg3: PulseConfig | None = None) -> list[PulseConfig]:
    if isinstance(pulse_cfg1, PulseConfig):
        configs = [pulse_cfg1]
    else:
        configs = list(pulse_cfg1)
    configs.extend(cfg for cfg in (pulse_cfg2, pulse_cfg3) if cfg is not None)

    if not 1 <= len(configs) <= 3:
        raise ValueError('Rabi_3Photon requires between one and three tones.')
    if not all(isinstance(cfg, PulseConfig) for cfg in configs):
        raise TypeError('Every tone must be a PulseConfig.')
    return configs


def _carrier_frequency(qubit: IdealGridium, pulse_cfg: PulseConfig) -> float:
    initial, final = pulse_cfg.targeted_drive
    base_frequency = pulse_cfg.drive_frequency
    if base_frequency is None:
        base_frequency = abs(qubit.freq(initial, final))
    carrier = float(base_frequency) + float(pulse_cfg.drive_detuning)
    if not np.isfinite(carrier) or carrier < 0:
        raise ValueError('The carrier frequency must be finite and non-negative.')
    return carrier


def _pulse_to_drive_term(
        qubit: IdealGridium, pulse_cfg: PulseConfig, phi_operator: qt.Qobj
        ) -> dict:
    if pulse_cfg.drive_type != 'flux':
        raise ValueError(
            "Rabi_3Photon currently supports only drive_type='flux'.")

    initial, final = pulse_cfg.targeted_drive
    if not (0 <= initial < qubit.nlev and 0 <= final < qubit.nlev):
        raise ValueError('targeted_drive levels must be within qubit.nlev.')
    matrix_element = abs(phi_operator[initial, final])
    if np.isclose(matrix_element, 0.0):
        raise ValueError(
            'The targeted transition has a zero flux matrix element; '
            'its drive normalization is undefined.')

    # Preserve the prior workflow convention: normalize phi to the selected
    # transition and apply drive_amplitude_factor as an operator scale.
    drive_term = {
        'operator': phi_operator / matrix_element,
        'amplitude': pulse_cfg.drive_amplitude_factor,
        'omega_d': _carrier_frequency(qubit, pulse_cfg),
        'phi': pulse_cfg.carrier_phase,
        'shape': pulse_cfg.pulse_shape,
        'sigma': pulse_cfg.pulse_sigma,
        'T_start': pulse_cfg.T_start,
        'T_gate': pulse_cfg.T_gate,
        'DRAG': pulse_cfg.DRAG,
        'DRAG_coefficient': pulse_cfg.DRAG_coeff,
    }
    if pulse_cfg.T_rise is not None:
        drive_term['T_rise'] = pulse_cfg.T_rise
    return drive_term


def logical_gate_target(target: str) -> qt.Qobj:
    """Return the requested target in the ordered logical basis ``|0>, |1>``.

    ``X180`` is Pauli X. ``X90`` is ``exp(-i*pi*X/4)``. Their physically
    irrelevant global phases are ignored by :func:`evaluate_logical_gate`.
    """
    target_name = target.upper()
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    if target_name == 'X180':
        matrix = pauli_x
    elif target_name == 'X90':
        matrix = (np.eye(2) - 1j * pauli_x) / np.sqrt(2)
    else:
        raise ValueError("target must be either 'X180' or 'X90'.")
    return qt.Qobj(matrix)


def evaluate_logical_gate(U_t, target: str, t_points=None) -> dict:
    """Evaluate logical action and leakage for a propagator trajectory.

    The final logical action is the 2-by-2 block ``A = P U(T) P`` in the
    ordered basis ``|0>, |1>``. Logical process fidelity is the normalized,
    phase-insensitive Hilbert--Schmidt overlap

    ``|Tr(V.dag() A)|^2 / (2 Tr(A.dag() A))``,

    where ``V`` is the requested target. Logical gate fidelity is its standard
    single-qubit Haar-average conversion ``(2 F_process + 1) / 3``. These
    condition out uniform loss from the logical block so gate-shape quality
    and leakage remain separate. Both values are defined as zero when the
    final logical block has zero norm.

    Leakage for input ``|j>`` is the explicitly summed population in levels
    2 and above. Aggregate leakage is the arithmetic mean for inputs ``|0>``
    and ``|1>``. Final and peak aggregate leakage are both returned, together
    with state-resolved population and leakage trajectories.
    """
    propagators = [U_t] if isinstance(U_t, qt.Qobj) else list(U_t)
    if not propagators:
        raise ValueError('U_t must contain at least one propagator.')

    matrices = []
    dimension = None
    for propagator in propagators:
        matrix = (
            propagator.full() if isinstance(propagator, qt.Qobj)
            else np.asarray(propagator, dtype=complex))
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Every propagator must be a square matrix.')
        if matrix.shape[0] < 2:
            raise ValueError('Every propagator must contain levels |0> and |1>.')
        if dimension is None:
            dimension = matrix.shape[0]
        elif matrix.shape != (dimension, dimension):
            raise ValueError('All propagators must have the same dimension.')
        matrices.append(matrix)

    if t_points is not None:
        times = np.asarray(t_points, dtype=float)
        if times.ndim != 1 or len(times) != len(matrices):
            raise ValueError('t_points must match the propagator trajectory.')
    else:
        times = None

    target_operator = logical_gate_target(target)
    target_matrix = target_operator.full()
    logical_action = matrices[-1][:2, :2]
    logical_norm = float(np.trace(
        logical_action.conj().T @ logical_action).real)
    if np.isclose(logical_norm, 0.0):
        logical_process_fidelity = 0.0
        logical_fidelity = 0.0
    else:
        overlap = np.trace(target_matrix.conj().T @ logical_action)
        logical_process_fidelity = float(
            abs(overlap) ** 2 / (2 * logical_norm))
        logical_process_fidelity = float(np.clip(
            logical_process_fidelity, 0.0, 1.0))
        logical_fidelity = (2 * logical_process_fidelity + 1) / 3

    state_diagnostics = {}
    state_leakage = []
    for initial in (0, 1):
        logical_populations = np.asarray([
            np.abs(matrix[:2, initial]) ** 2 for matrix in matrices])
        leakage = np.asarray([
            np.sum(np.abs(matrix[2:, initial]) ** 2)
            for matrix in matrices], dtype=float)
        total_probability = np.asarray([
            np.sum(np.abs(matrix[:, initial]) ** 2)
            for matrix in matrices], dtype=float)
        peak_index = int(np.argmax(leakage))
        state_diagnostics[initial] = {
            'logical_populations': logical_populations,
            'leakage': leakage,
            'total_probability': total_probability,
            'final_leakage': float(leakage[-1]),
            'peak_leakage': float(leakage[peak_index]),
            'peak_leakage_index': peak_index,
            'peak_leakage_time': (
                None if times is None else float(times[peak_index])),
        }
        state_leakage.append(leakage)

    leakage_by_time = np.mean(np.asarray(state_leakage), axis=0)
    peak_index = int(np.argmax(leakage_by_time))
    return {
        'target': target.upper(),
        'target_operator': target_operator,
        'logical_action': qt.Qobj(logical_action),
        'logical_process_fidelity': logical_process_fidelity,
        'logical_gate_fidelity': logical_fidelity,
        'leakage_by_time': leakage_by_time,
        'final_leakage': float(leakage_by_time[-1]),
        'peak_leakage': float(leakage_by_time[peak_index]),
        'peak_leakage_index': peak_index,
        'peak_leakage_time': (
            None if times is None else float(times[peak_index])),
        'state_diagnostics': state_diagnostics,
    }


def state_qubit_state(
        qubit: IdealGridium,
        pulse_cfg1: PulseConfig | Sequence[PulseConfig],
        pulse_cfg2: PulseConfig | None = None,
        pulse_cfg3: PulseConfig | None = None):
    """Print the targeted transition and resolved carrier for each tone."""
    _require_idealgridium(qubit)
    configs = _coerce_pulse_configs(pulse_cfg1, pulse_cfg2, pulse_cfg3)
    for index, config in enumerate(configs, start=1):
        initial, final = config.targeted_drive
        print(
            'Tone {} targeting transition {} to {}: {:.6f} GHz; '
            'detuning {:.6f} GHz; carrier {:.6f} GHz'.format(
                index, initial, final, abs(qubit.freq(initial, final)),
                config.drive_detuning, _carrier_frequency(qubit, config)))


def solve(
        qubit: IdealGridium,
        pulse_cfg1: PulseConfig | Sequence[PulseConfig],
        pulse_cfg2: PulseConfig | None = None,
        pulse_cfg3: PulseConfig | None = None,
        comp_space: Sequence[int] = (0, 1),
        solve_method: str = 'propagator',
        mute: bool = False):
    """Execute one, two, or three independent IdealGridium flux tones."""
    del comp_space  # Retained only for compatibility with the old entry point.
    _require_idealgridium(qubit)
    if solve_method != 'propagator':
        raise ValueError("Only solve_method='propagator' is implemented.")

    configs = _coerce_pulse_configs(pulse_cfg1, pulse_cfg2, pulse_cfg3)
    phi_operator = qubit.phi()
    drive_terms = [
        _pulse_to_drive_term(qubit, config, phi_operator)
        for config in configs
    ]
    final_time = max(config.T_start + config.T_gate for config in configs)
    t_points = np.linspace(0, final_time, 2 * int(final_time) + 1)
    U_t = gates.evolution_operator_multitone_microwave(
        qubit.H(), drive_terms, t_points=t_points)

    if not mute:
        state_qubit_state(qubit, configs)
    return t_points, U_t


def visualize_state_propagation(
        qubit: IdealGridium,
        pulse_configs: Sequence[PulseConfig],
        t_points,
        U_t,
        n_shown_states: int = 3,
        comp_space: Sequence[int] = (0, 1)):
    """Plot the most populated output levels for selected initial states."""
    configs = _coerce_pulse_configs(pulse_configs)
    fig, axes = plt.subplots(
        1, len(comp_space), figsize=(6 * len(comp_space), 5), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        '{} with {}'.format(
            qubit.name,
            ', '.join(
                'tone {} driving {}'.format(index, cfg.targeted_drive)
                for index, cfg in enumerate(configs, start=1))))

    for ax, initial in zip(axes, comp_space):
        probabilities = {
            final: gates.prob_transition(
                U_t, qt.basis(qubit.nlev, initial),
                qt.basis(qubit.nlev, final))
            for final in range(qubit.nlev)
        }
        shown = sorted(
            probabilities,
            key=lambda final: np.max(probabilities[final]),
            reverse=True)[:n_shown_states]
        for final in shown:
            ax.plot(
                t_points, probabilities[final],
                label=r'$P({}\rightarrow{})$'.format(initial, final))
        ax.legend(loc='best')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'$P_{i\rightarrow f}$')
        ax.set_title(r'Starting in $|{}\rangle$'.format(initial))
    fig.tight_layout()
    return fig


def solve_multitone_drive(
        qubit: IdealGridium,
        pulse_paths: Iterable[str | Path],
        qubit_cache_directory: str | Path | None = None,
        mute: bool = False):
    """Load and execute one to three pulse YAML files."""
    pulse_configs = [load_pulse_config(path) for path in pulse_paths]
    qubit = load_qubit(qubit, qubit_cache_directory)
    return solve(qubit, pulse_configs, solve_method='propagator', mute=mute)


def solve_two_photon_drive(
        qubit: IdealGridium, pulse_path1, pulse_path2,
        n_shown_states: int = 3,
        qubit_cache_directory: str | Path | None = None):
    """Compatibility plotting wrapper for the existing two-file workflow."""
    configs = [load_pulse_config(pulse_path1), load_pulse_config(pulse_path2)]
    qubit = load_qubit(qubit, qubit_cache_directory)
    t_points, U_t = solve(qubit, configs)
    return visualize_state_propagation(
        qubit, configs, t_points, U_t, n_shown_states=n_shown_states)


def solve_three_tone_drive(
        qubit: IdealGridium, pulse_path1, pulse_path2, pulse_path3,
        qubit_cache_directory: str | Path | None = None,
        mute: bool = False):
    """Load and execute exactly three independently configured tones."""
    return solve_multitone_drive(
        qubit, [pulse_path1, pulse_path2, pulse_path3],
        qubit_cache_directory=qubit_cache_directory, mute=mute)
