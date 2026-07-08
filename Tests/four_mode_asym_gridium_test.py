# Tests for the exact four-mode asymmetric-KITE gridium (qchard_four_mode_asym_gridium).
#
# The heavy physics validation (converged doublet degeneracies, full spectra vs flux,
# and ng-periodicity dispersion) requires production cutoffs and minutes/hours of runtime;
# those live in the simulation notebooks/scripts. Here we pin the implementation and the
# lightweight validation plumbing:
#
# 1. The two-stage hierarchical diagonalization, when keeping the FULL first-sector basis,
#    must agree with a brute-force diagonalization of the full four-mode tensor-product
#    Hamiltonian to numerical precision (it is then an exact unitary rearrangement).
#    This is checked with all asymmetry knobs on (eps_J, eps_LK, ng, phi_ext != 0).

# 2. Hermiticity/finiteness of spectra and drive operators at representative smoke cases.
# 3. Symmetry selection rules that should hold exactly at the symmetric protection point.
# 4. Truncation-probe bookkeeping at tiny cutoffs; this is not a convergence certificate.

import inspect
import sys
import os
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.sparse as sps
import pytest

from Circuit_Objs.qchard_four_mode_asym_gridium import (
    FourModeAsymGridium, device_FourModeAsymGridium_params,
    std_FourModeAsymGridium_sim_params)

TINY = dict(n_charge=2, n_grid_pts=31, grid_range=8.0, nlev_delta=4, nlev_node=4)
S1DIM = (2 * TINY['n_charge'] + 1) * TINY['n_grid_pts'] * TINY['nlev_delta']
SMOKE = dict(n_charge=3, n_grid_pts=61, grid_range=8.0,
             nlev_delta=5, nlev_node=5, nkeep_s1=24)
PROBE_BASE = dict(n_charge=2, n_grid_pts=31, grid_range=8.0,
                  nlev_delta=4, nlev_node=4, nkeep_s1=12)
REPRESENTATIVE_CASES = [
    ('symmetric_protection',
     dict(eps_J=0.0, eps_LK=0.0, ng=0.0, phi_ext=0.0, theta_ext=np.pi)),
    ('junction_asym_protection',
     dict(eps_J=0.07, eps_LK=0.0, ng=0.0, phi_ext=0.0, theta_ext=np.pi)),
    ('mixed_asym_offset',
     dict(eps_J=0.07, eps_LK=0.04, ng=0.3, phi_ext=0.2, theta_ext=np.pi + 0.1)),
]
TRUNCATION_DOUBLERS = {
    'n_charge': lambda v: 2 * v,
    'n_grid_pts': lambda v: 2 * v - 1,
    'grid_range': lambda v: 1.5 * v,
    'nlev_delta': lambda v: 2 * v,
    'nlev_node': lambda v: 2 * v,
    'nkeep_s1': lambda v: 2 * v,
}


def _make_qubit(nlev=8, cutoffs=None, **params):
    opts = dict(std_FourModeAsymGridium_sim_params)
    if cutoffs is not None:
        opts.update(cutoffs)
    opts.update(params)
    opts['nlev'] = nlev
    return FourModeAsymGridium(**device_FourModeAsymGridium_params, **opts)


def _block_snorm(M, rows, cols):
    return float(np.linalg.svd(M[np.ix_(rows, cols)], compute_uv=False)[0])


def _traceless_block_norm(M, rows):
    B = M[np.ix_(rows, rows)]
    B = B - np.trace(B) * np.eye(len(rows)) / len(rows)
    return float(np.linalg.norm(B, ord='fro'))


def _observables(q):
    d = q.doublet_splittings()
    phi2 = q.phase_grid(nlev=6).full()
    dth = q.d_theta(nlev=6).full()
    return {
        'd01_MHz': 1e3 * d[0],
        'd23_MHz': 1e3 * d[1],
        'd45_MHz': 1e3 * d[2],
        '|phi2_D0|': _traceless_block_norm(phi2, [0, 1]),
        '|phi2_D2|': _traceless_block_norm(phi2, [4, 5]),
        '||D0_phi2_D2||': _block_snorm(phi2, [0, 1], [4, 5]),
        '|Dth_D0|': _traceless_block_norm(dth, [0, 1]),
        '|Dth_D2|': _traceless_block_norm(dth, [4, 5]),
        '||D0_Dth_D2||': _block_snorm(dth, [0, 1], [4, 5]),
    }


def _brute_force_levels(q, nlev):

    """Full four-mode tensor-product diagonalization with the same node-sector convention
    (analytic oscillator ladder + explicit couplings) as the class."""
    P = device_FourModeAsymGridium_params
    o = q._mode_ops()

    def K4(A, B, C, D):
        return sps.kron(sps.kron(sps.kron(sps.csr_matrix(A), sps.csr_matrix(B)),
                                 sps.csr_matrix(C)), sps.csr_matrix(D)).tocsr()

    I1, Ig, I3 = o['I1'], o['Ig'], o['I3']
    I4n = np.eye(q.nlev_node)
    ELK, EL, EJ = P['E_LK'], P['E_L'], P['E_J']
    EJS, ECS, EC, ECD = P['E_JS'], P['E_CS'], P['E_C'], P['E_C_delta']
    eJ, eLK, ng, u40 = q.eps_J, q.eps_LK, q.ng, o['u40']

    pg = (K4(o['m1'], Ig, I3, I4n) + K4(I1, o['m2'], I3, I4n) + ng * K4(I1, Ig, I3, I4n))
    H = (4 * ECS * (pg @ pg)
         + 2 * EC * K4(I1, o['m2sq'], I3, I4n)
         + 2 * ECD * K4(I1, Ig, o['m3'] @ o['m3'], I4n)
         - EJS * K4(o['cos1'], Ig, I3, I4n)
         + ELK * K4(I1, o['x2d'], I3, I4n)
         + ELK * K4(I1, Ig, o['v3'] @ o['v3'], I4n)
         - 2 * EJ * (K4(o['cos1'], o['cxd'], o['cos_u3'], I4n)
                     + K4(o['sin1'], o['sxd'], o['cos_u3'], I4n))
         - 2 * eJ * EJ * (K4(o['sin1'], o['cxd'], o['sin_u3'], I4n)
                          - K4(o['cos1'], o['sxd'], o['sin_u3'], I4n))
         + 2 * eLK * ELK * K4(I1, o['xd'], o['v3'], I4n)
         - 2 * ELK * u40 * K4(I1, o['xd'], I3, I4n)
         - 2 * eLK * ELK * u40 * K4(I1, Ig, o['v3'], I4n)
         + K4(I1, Ig, I3, np.diag(o['w4']))
         - 2 * ELK * K4(I1, o['xd'], I3, o['v4'])
         - 2 * eLK * ELK * K4(I1, Ig, o['v3'], o['v4'])
         + 8 * ECS * (pg @ K4(I1, Ig, I3, o['m4'])))
    Hd = 0.5 * (H + H.conj().T).toarray()
    return np.linalg.eigvalsh(Hd)[:nlev]

#checks the default values of the constructor match the shared dictionary 
def test_constructor_defaults_match_standard_sim_params():
    sig = inspect.signature(FourModeAsymGridium)
    for key, expected in std_FourModeAsymGridium_sim_params.items():
        got = sig.parameters[key].default
        if isinstance(expected, float):
            assert np.isclose(got, expected)
        else:
            assert got == expected


@pytest.mark.parametrize('bad_kw', [
    {'nlev': 0},
    {'n_charge': 0},
    {'n_grid_pts': 4},
    {'grid_range': 0.0},
    {'nlev_delta': 0},
    {'nlev_node': 0},
    {'nkeep_s1': 0},
])

#checks for invalid numerical cutoffs 
def test_invalid_truncation_inputs_are_rejected(bad_kw):
    opts = dict(std_FourModeAsymGridium_sim_params)
    opts.update(bad_kw)
    with pytest.raises(ValueError):
        FourModeAsymGridium(**device_FourModeAsymGridium_params, **opts)


@pytest.mark.parametrize('eps_J,eps_LK,ng,phi_ext', [
    (0.0, 0.0, 0.0, 0.0),
    (0.07, 0.04, 0.3, 0.2),
])
#checks that the two-stage hierarchial solver gives the same low-energy spectrum as a direct four mode diagonalization
def test_hierarchy_matches_brute_force(eps_J, eps_LK, ng, phi_ext):
    q = FourModeAsymGridium(**device_FourModeAsymGridium_params,
                            eps_J=eps_J, eps_LK=eps_LK, ng=ng, phi_ext=phi_ext,
                            nlev=10, nkeep_s1=S1DIM, **TINY)
    E_hier = q.levels(nlev=10)
    E_brute = _brute_force_levels(q, 10)
    assert np.max(np.abs(E_hier - E_brute)) < 1e-9


@pytest.mark.parametrize('name,case', REPRESENTATIVE_CASES)
#checks at the following points that nothing breaks: 
#1. symmetric protection point: 
#   eps_J = 0.0
#   eps_LK = 0.0
#   ng = 0.0
#   phi_ext = 0.0
#   theta_ext = np.pi
#2. junction asymmetry only point: 
#   eps_J = 0.07
#   eps_LK = 0.0
#   ng = 0.0
#   phi_ext = 0.0
#   theta_ext = np.pi
#3. mixed asymmetry and offset point: 
#   eps_J = 0.07
#   eps_LK = 0.04
#   ng = 0.3
#   phi_ext = 0.2
#   theta_ext = np.pi + 0.1

def test_finite_sanity_for_representative_smoke_cases(name, case):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        q = _make_qubit(cutoffs=SMOKE, **case)
        levels = np.real(q.levels(nlev=8))
        ops = {
            'd_theta': q.d_theta(8).full(),
            'd_phi': q.d_phi(8).full(),
            'phase_grid': q.phase_grid(8).full(),
            'n_cap': q.n_cap(8).full(),
        }

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], name
    assert np.all(np.isfinite(levels)), name
    for op_name, M in ops.items():
        assert np.all(np.isfinite(M)), (name, op_name)
        assert np.max(np.abs(M - M.conj().T)) < 1e-9, (name, op_name)

#checks the operators are hermitian: d_theta, d_phi, phase_grid, n_cap
#checks methods are working correctly 
def test_operators_hermitian_and_api():
    q = FourModeAsymGridium(**device_FourModeAsymGridium_params,
                            eps_J=0.05, eps_LK=0.05, ng=0.1,
                            nlev=8, nkeep_s1=40, **TINY)
    for op in [q.d_theta(), q.d_phi(), q.phase_grid(), q.n_cap()]:
        M = op.full()
        assert np.max(np.abs(M - M.conj().T)) < 1e-10
    # duck-typed API used by the gate workflow
    assert q.phi().shape == (8, 8)
    assert q.phase_grid().shape == (8, 8)
    assert abs(q.freq(0, 1) - (q.level(1) - q.level(0))) < 1e-12
    assert len(q.doublet_splittings()) == 3
    # cache invalidation on parameter change
    E_before = q.level(1)
    q.eps_J = 0.10
    assert q._eigvals is None
    assert q.level(1) != E_before

#checks at the following symmetric protection points: 
#eps_J = 0
#eps_LK = 0
#ng = 0
#phi_ext = 0
#theta_ext = np.pi
#the operators are effectivly dark 
def test_symmetric_dark_channels_at_protection_bias():
    q = _make_qubit(cutoffs=SMOKE, eps_J=0.0, eps_LK=0.0, ng=0.0,
                    phi_ext=0.0, theta_ext=np.pi)

    d_phi = q.d_phi(8).full()
    n_cap = q.n_cap(8).full()
    phase_grid = q.phase_grid(8).full()
    d_theta = q.d_theta(8).full()

    assert _traceless_block_norm(d_phi, [0, 1]) < 1e-10
    assert _traceless_block_norm(n_cap, [0, 1]) < 1e-10
    assert _traceless_block_norm(phase_grid, [0, 1]) < 1e-10
    assert _traceless_block_norm(d_theta, [0, 1]) > 1e-3

#checks small cutoff-sensitivity probe runs correctly 
#doubles one cutoff at a time
#recomputes the same observables
#checks that all values are finite
def test_tiny_truncation_probe_bookkeeping():
    base = _observables(_make_qubit(cutoffs=PROBE_BASE))

    assert set(TRUNCATION_DOUBLERS) == {
        'n_charge', 'n_grid_pts', 'grid_range',
        'nlev_delta', 'nlev_node', 'nkeep_s1',
    }
    assert all(np.isfinite(v) for v in base.values())

    for knob, double in TRUNCATION_DOUBLERS.items():
        cutoffs = dict(PROBE_BASE)
        cutoffs[knob] = double(cutoffs[knob])
        obs = _observables(_make_qubit(cutoffs=cutoffs))
        rel = {key: abs(obs[key] - base[key]) / max(abs(base[key]), 1e-12)
               for key in base}

        assert all(np.isfinite(v) for v in obs.values()), knob
        assert set(rel) == set(base), knob
        assert all(np.isfinite(v) for v in rel.values()), knob
