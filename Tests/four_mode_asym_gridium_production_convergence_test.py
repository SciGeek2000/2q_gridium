"""Opt-in production convergence sweeps for the four-mode asymmetric gridium.

These tests intentionally do not run during normal pytest invocations. Set
RUN_FOUR_MODE_PRODUCTION=1 to execute the full production sweeps. Results are
cached as JSON under FOUR_MODE_PRODUCTION_OUTPUT_DIR, or under /private/tmp by
default, so interrupted long runs can be resumed.

The default assertions check that every completed sweep point returns finite,
Hermitian low-energy data. Set FOUR_MODE_REQUIRE_CONVERGENCE=1 to additionally
require the one-at-a-time production basis-doubling comparisons to satisfy the
current 1% convergence criterion.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from Circuit_Objs.qchard_four_mode_asym_gridium import (
    FourModeAsymGridium,
    device_FourModeAsymGridium_params,
    std_FourModeAsymGridium_sim_params,
)


RUN_PRODUCTION = os.environ.get('RUN_FOUR_MODE_PRODUCTION', '').lower() in {'1', 'true', 'yes'}
FORCE = os.environ.get('FOUR_MODE_PRODUCTION_FORCE', '').lower() in {'1', 'true', 'yes'}
ALLOW_WARNINGS = os.environ.get('FOUR_MODE_ALLOW_RUNTIME_WARNINGS', '').lower() in {'1', 'true', 'yes'}
REQUIRE_CONVERGENCE = os.environ.get('FOUR_MODE_REQUIRE_CONVERGENCE', '').lower() in {
    '1', 'true', 'yes'
}
REL_TOL = float(os.environ.get('FOUR_MODE_PRODUCTION_REL_TOL', '0.01'))

pytestmark = pytest.mark.skipif(
    not RUN_PRODUCTION,
    reason='Set RUN_FOUR_MODE_PRODUCTION=1 to run hour-scale production convergence sweeps.',
)


CASE_NAME = 'symmetric_protection'
CASE_PARAMS = dict(eps_J=0.0, eps_LK=0.0, ng=0.0, phi_ext=0.0, theta_ext=np.pi)

DEFAULT_KNOBS = ('nkeep_s1', 'n_charge', 'n_grid_pts', 'grid_range', 'nlev_delta', 'nlev_node')
DEFAULT_NKEEP_LADDER = (80, 160, 320, 640)
HIGH_CUTOFF_POINTS = (
    (20, 24, 640),
    (30, 24, 640),
    (30, 32, 640),
    (30, 32, 1000),
)
HIGH_CUTOFF_BASE = dict(n_charge=8, n_grid_pts=181, grid_range=12.0, nlev=8)

ABS_TOL = {
    'd01_MHz': 0.1,
    'd23_MHz': 0.1,
    'd45_MHz': 0.1,
    '|phi2_D0|': 1e-3,
    '|phi2_D2|': 1e-3,
    '||D0_phi2_D2||': 1e-3,
    '|Dth_D0|': 1e-3,
    '|Dth_D2|': 1e-3,
    '||D0_Dth_D2||': 1e-3,
}


def _output_dir():
    default = Path('/private/tmp/four_mode_asym_gridium_production_convergence')
    return Path(os.environ.get('FOUR_MODE_PRODUCTION_OUTPUT_DIR', default))


def _json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Object of type %s is not JSON serializable' % type(obj).__name__)


def _read_json(path):
    if path.exists() and not FORCE:
        return json.loads(path.read_text())
    return None


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + '\n')


def _double_value(knob, value):
    if knob == 'n_grid_pts':
        return 2 * value - 1
    if knob == 'grid_range':
        return 1.5 * value
    return 2 * value


def _block_snorm(M, rows, cols):
    return float(np.linalg.svd(M[np.ix_(rows, cols)], compute_uv=False)[0])


def _traceless_block_norm(M, rows):
    B = M[np.ix_(rows, rows)]
    B = B - np.trace(B) * np.eye(len(rows)) / len(rows)
    return float(np.linalg.norm(B, ord='fro'))


def _observables(qubit):
    d = qubit.doublet_splittings()
    phi2 = qubit.phase_grid(nlev=6).full()
    dth = qubit.d_theta(nlev=6).full()
    return {
        'd01_MHz': float(1e3 * d[0]),
        'd23_MHz': float(1e3 * d[1]),
        'd45_MHz': float(1e3 * d[2]),
        '|phi2_D0|': _traceless_block_norm(phi2, [0, 1]),
        '|phi2_D2|': _traceless_block_norm(phi2, [4, 5]),
        '||D0_phi2_D2||': _block_snorm(phi2, [0, 1], [4, 5]),
        '|Dth_D0|': _traceless_block_norm(dth, [0, 1]),
        '|Dth_D2|': _traceless_block_norm(dth, [4, 5]),
        '||D0_Dth_D2||': _block_snorm(dth, [0, 1], [4, 5]),
    }


def _make_qubit(case_params, overrides):
    params = dict(case_params)
    eps_J = params.pop('eps_J', 0.0)
    eps_LK = params.pop('eps_LK', 0.0)
    ng = params.pop('ng', 0.0)

    sim = dict(std_FourModeAsymGridium_sim_params)
    sim.update(overrides)
    sim.update(params)
    sim.update(eps_J=eps_J, eps_LK=eps_LK, ng=ng)
    return FourModeAsymGridium(**device_FourModeAsymGridium_params, **sim)


def _compute_snapshot(case_params, overrides, nlev=8, op_nlev=8):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        q = _make_qubit(case_params, overrides)
        levels = np.real(q.levels(nlev=nlev))
        ops = {
            'd_theta': q.d_theta(op_nlev).full(),
            'd_phi': q.d_phi(op_nlev).full(),
            'phase_grid': q.phase_grid(op_nlev).full(),
            'n_cap': q.n_cap(op_nlev).full(),
        }
        obs = _observables(q)

    finite_ops = {key: bool(np.all(np.isfinite(value))) for key, value in ops.items()}
    herm = {key: float(np.max(np.abs(value - value.conj().T))) for key, value in ops.items()}
    return {
        'case': CASE_NAME,
        'case_params': dict(case_params),
        'overrides': dict(overrides),
        'levels_GHz': levels.tolist(),
        'transition_GHz': (levels - levels[0]).tolist(),
        'observables': obs,
        'finite_levels': bool(np.all(np.isfinite(levels))),
        'finite_ops': finite_ops,
        'hermiticity_error': herm,
        'runtime_warnings': sorted(set(str(w.message) for w in caught)),
        'seconds': time.time() - t0,
    }


def _snapshot(cache_name, case_params, overrides, nlev=8, op_nlev=8):
    path = _output_dir() / (cache_name + '.json')
    cached = _read_json(path)
    if cached is not None:
        return cached

    snap = _compute_snapshot(case_params, overrides, nlev=nlev, op_nlev=op_nlev)
    _write_json(path, snap)
    return snap


def _compare(base, variant, rel_tol):
    out = {}
    for key, base_val in base.items():
        var_val = variant[key]
        abs_delta = abs(var_val - base_val)
        scale = abs(base_val)
        rel_delta = abs_delta / max(scale, 1e-12)
        abs_tol = ABS_TOL.get(key, 1e-3)
        if scale < abs_tol:
            passed = abs_delta < abs_tol
            criterion = 'absolute'
        else:
            passed = rel_delta < rel_tol
            criterion = 'relative'
        out[key] = {
            'base': base_val,
            'variant': var_val,
            'absolute_change': abs_delta,
            'relative_change': rel_delta,
            'absolute_tolerance': abs_tol,
            'relative_tolerance': rel_tol,
            'criterion': criterion,
            'passed': bool(passed),
        }

    worst_rel = max(out, key=lambda k: out[k]['relative_change'])
    worst_abs = max(out, key=lambda k: out[k]['absolute_change'])
    failed = [key for key, value in out.items() if not value['passed']]
    return {
        'by_observable': out,
        'worst_relative_observable': worst_rel,
        'worst_relative_change': out[worst_rel]['relative_change'],
        'worst_absolute_observable': worst_abs,
        'worst_absolute_change': out[worst_abs]['absolute_change'],
        'failed_observables': failed,
        'passed': len(failed) == 0,
    }


def _assert_snapshot_healthy(snapshot):
    assert snapshot['finite_levels']
    assert all(snapshot['finite_ops'].values())
    assert max(snapshot['hermiticity_error'].values()) < 1e-8
    assert np.all(np.isfinite(np.asarray(snapshot['levels_GHz'], dtype=float)))
    assert all(np.isfinite(value) for value in snapshot['observables'].values())
    if not ALLOW_WARNINGS:
        assert snapshot['runtime_warnings'] == []

#runs full simulation at the symmetri protection and point and checks:
#   - all energy levels are finite 
#   - all operators are finite 
#   - all operators are hermitian 
#   - no unexpected runtime warnings
#across several values of nkeep_s1: 80, 160, 320, 640
@pytest.mark.parametrize('nkeep_s1', DEFAULT_NKEEP_LADDER)
def test_production_nkeep_ladder_point(nkeep_s1):
    snap = _snapshot(
        cache_name='production_nkeep_%d' % nkeep_s1,
        case_params=CASE_PARAMS,
        overrides={'nkeep_s1': int(nkeep_s1)},
        nlev=8,
        op_nlev=8,
    )
    _assert_snapshot_healthy(snap)

#starts from the standard production cutoffs, doubles one numerical basis cutoff at a time: 
#   nkeep_s1
#   n_charge
#   n_grid_pts
#   grid_range
#   nlev_delta
#   nlev_node
#compares the physical observables before and after the change for convergence
#saves the comparison to JSON
@pytest.mark.parametrize('knob', DEFAULT_KNOBS)
def test_production_basis_doubling_probe(knob):
    baseline = _snapshot(
        cache_name='production_basis_baseline',
        case_params=CASE_PARAMS,
        overrides={},
        nlev=8,
        op_nlev=8,
    )
    base_value = std_FourModeAsymGridium_sim_params[knob]
    variant_overrides = {knob: _double_value(knob, base_value)}
    variant = _snapshot(
        cache_name='production_basis_%s' % knob,
        case_params=CASE_PARAMS,
        overrides=variant_overrides,
        nlev=8,
        op_nlev=8,
    )

    _assert_snapshot_healthy(baseline)
    _assert_snapshot_healthy(variant)

    comparison = _compare(baseline['observables'], variant['observables'], REL_TOL)
    report_path = _output_dir() / 'production_basis_comparisons.json'
    report = _read_json(report_path) or {}
    report[knob] = {
        'baseline': baseline['observables'],
        'variant_overrides': variant_overrides,
        'variant': variant['observables'],
        'comparison': comparison,
    }
    _write_json(report_path, report)

    if REQUIRE_CONVERGENCE:
        assert comparison['passed'], comparison


# for (nlev_delta, nlev_node, nkeep_s1) at sveral large cutoff points: 
#(20, 24, 640)
#(30, 24, 640)
#(30, 32, 640)
#(30, 32, 1000)
#checks that the spectrum and projected operators are finite, Hermitian, and free of unexpected runtime warnings.
@pytest.mark.parametrize('nlev_delta,nlev_node,nkeep_s1', HIGH_CUTOFF_POINTS)
def test_high_cutoff_convergence_ladder_point(nlev_delta, nlev_node, nkeep_s1):
    overrides = dict(HIGH_CUTOFF_BASE)
    overrides.update(
        nlev_delta=int(nlev_delta),
        nlev_node=int(nlev_node),
        nkeep_s1=int(nkeep_s1),
    )
    snap = _snapshot(
        cache_name='high_cutoff_delta%d_node%d_nkeep%d' % (nlev_delta, nlev_node, nkeep_s1),
        case_params=CASE_PARAMS,
        overrides=overrides,
        nlev=8,
        op_nlev=6,
    )
    _assert_snapshot_healthy(snap)

#compares each high-cutoff result with the next one: 
#(20,24,640) → (30,24,640)
#(30,24,640) → (30,32,640)
#(30,32,640) → (30,32,1000)
#and checks how much observable changes, saves the comparison to JSON
def test_high_cutoff_ladder_comparison_bookkeeping():
    points = []
    for nlev_delta, nlev_node, nkeep_s1 in HIGH_CUTOFF_POINTS:
        cache_name = 'high_cutoff_delta%d_node%d_nkeep%d' % (nlev_delta, nlev_node, nkeep_s1)
        overrides = dict(HIGH_CUTOFF_BASE)
        overrides.update(
            nlev_delta=int(nlev_delta),
            nlev_node=int(nlev_node),
            nkeep_s1=int(nkeep_s1),
        )
        points.append(
            _snapshot(
                cache_name=cache_name,
                case_params=CASE_PARAMS,
                overrides=overrides,
                nlev=8,
                op_nlev=6,
            )
        )

    comparisons = {}
    for previous, current in zip(points[:-1], points[1:]):
        prev_key = 'delta{nlev_delta}_node{nlev_node}_nkeep{nkeep_s1}'.format(
            **previous['overrides']
        )
        cur_key = 'delta{nlev_delta}_node{nlev_node}_nkeep{nkeep_s1}'.format(
            **current['overrides']
        )
        comparisons[prev_key + '_to_' + cur_key] = _compare(
            previous['observables'],
            current['observables'],
            REL_TOL,
        )

    _write_json(_output_dir() / 'high_cutoff_ladder_comparisons.json', comparisons)
    assert set(comparisons) == {
        'delta20_node24_nkeep640_to_delta30_node24_nkeep640',
        'delta30_node24_nkeep640_to_delta30_node32_nkeep640',
        'delta30_node32_nkeep640_to_delta30_node32_nkeep1000',
    }
