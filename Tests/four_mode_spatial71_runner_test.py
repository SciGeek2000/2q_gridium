import json

import numpy as np
import pytest
import scipy.sparse as sps

from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner


def _write_complete_fixture(directory, configuration=None,
                            fingerprint=None):
    directory.mkdir()
    for name in (
            'stage1_eigensystem.npz',
            'projected_operators.npz',
            'analysis.json'):
        (directory / name).write_bytes(f'fixture:{name}'.encode())
    metadata = {
        'status': 'complete',
        'configuration': (
            runner.FROZEN_CONFIGURATION
            if configuration is None else configuration
        ),
        'configuration_sha256': (
            runner.CONFIGURATION_SHA256
            if fingerprint is None else fingerprint
        ),
        'file_sha256': {
            name: runner._sha256_file(directory / name)
            for name in (
                'stage1_eigensystem.npz',
                'projected_operators.npz',
                'analysis.json',
            )
        },
    }
    (directory / 'checkpoint.json').write_text(json.dumps(metadata))
    return metadata


def test_frozen_configuration_and_candidate_definitions():
    assert runner.EXPECTED_STAGE1_DIMENSION == 45369
    assert runner.configuration_fingerprint() == (
        '2c996f23c3f4810f9bb7d72f9231f264cda516900b4a7067af2ab53dbd1b868a'
    )
    selected = runner.retained_indices(170)
    assert selected.tolist() == (
        list(range(160))
        + [162, 163, 165, 167, 168, 170, 171, 172, 176, 177]
    )
    assert len(runner.retained_indices(172)) == 172
    assert len(runner.retained_indices(175)) == 175
    assert runner.retained_indices(172)[160:].tolist() == [
        160, 162, 163, 165, 166, 167, 168, 170, 171, 172, 176, 177,
    ]
    assert runner.retained_indices(175)[160:].tolist() == [
        160, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173,
        174, 176, 177,
    ]


def test_frozen_physics_and_spatial_parameters_reach_stage1(monkeypatch):
    coefficient_arguments = []
    assembly_arguments = []

    def fake_coefficients(*arguments):
        coefficient_arguments.append(arguments)
        return np.eye(4), np.eye(3), 4.5, 5.5

    def fake_assembly(*arguments):
        assembly_arguments.append(arguments)
        return sps.csr_matrix((45369, 45369)), {'sentinel': True}

    monkeypatch.setattr(runner, '_coeffs', fake_coefficients)
    monkeypatch.setattr(runner, '_assemble_nonlinear_sector', fake_assembly)
    _, _, matrix, raw = runner._assemble_stage1()

    assert coefficient_arguments == [(
        5.0, 0.5, 1.0, 1.0, 4.0, 8.0, 0.10, 0.05, 5.5, 10.0,
    )]
    routed = assembly_arguments[0]
    assert routed[2:] == (
        4.5, 5.5, 4.0, 0.0, 0.0, np.pi,
        4, 71, 11.0, 71, 14.0,
    )
    assert matrix.shape == (45369, 45369)
    assert raw == {'sentinel': True}


def test_frozen_solver_settings_reach_single_stage1_call(monkeypatch):
    calls = []

    monkeypatch.setattr(
        runner, '_assemble_stage1',
        lambda: (np.eye(4), np.eye(3), sps.eye(1), {}),
    )

    def fake_eigensolve(matrix, **settings):
        calls.append(settings)
        return np.arange(180.0), np.ones((1, 1)), {}

    monkeypatch.setattr(runner, '_stage1_eigensolve', fake_eigensolve)
    monkeypatch.setattr(
        runner, '_validate_eigensystem',
        lambda matrix, values, vectors: (np.zeros(180), 0.0),
    )
    monkeypatch.setattr(
        runner, '_project_operators',
        lambda raw, vectors: {name: np.zeros((1, 1))
                              for name in runner.EXPECTED_OPERATOR_NAMES},
    )

    runner._solve_stage1_once()
    assert calls == [{
        'k': 180,
        'sigma': -24.0,
        'which': 'LM',
        'tol': 1e-8,
        'ncv': None,
        'v0': None,
        'permc_spec': 'MMD_AT_PLUS_A',
        'collect_diagnostics': True,
    }]


def test_dry_run_never_assembles_hamiltonian(monkeypatch, tmp_path):
    def forbidden():
        raise AssertionError('dry-run assembled the Hamiltonian')

    monkeypatch.setattr(runner, '_assemble_stage1', forbidden)
    manifest = runner.dry_run_manifest(tmp_path, tmp_path / 'n51')
    assert manifest['mode'] == 'dry-run-no-matrix-assembly'
    assert manifest['configuration_sha256'] == runner.CONFIGURATION_SHA256
    assert manifest['expected_stage1_dimension'] == 45369


def test_configuration_and_thread_mismatches_fail_closed():
    changed = json.loads(json.dumps(runner.FROZEN_CONFIGURATION))
    changed['spatial_cutoffs']['N2'] = 69
    with pytest.raises(ValueError, match='does not exactly match'):
        runner.validate_frozen_configuration(changed)
    with pytest.raises(RuntimeError, match='OPENBLAS_NUM_THREADS=1'):
        runner.validate_thread_environment({
            'OPENBLAS_NUM_THREADS': '2',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        })


def test_every_frozen_parameter_contributes_to_fingerprint():
    original = runner.configuration_fingerprint()
    for section, settings in runner.FROZEN_CONFIGURATION.items():
        for name, value in settings.items():
            changed = json.loads(json.dumps(runner.FROZEN_CONFIGURATION))
            if value is None:
                replacement = 17
            elif isinstance(value, str):
                replacement = value + '_changed'
            else:
                replacement = value + 1
            changed[section][name] = replacement
            assert runner.configuration_fingerprint(changed) != original, (
                section, name
            )
            with pytest.raises(ValueError, match='does not exactly match'):
                runner.validate_frozen_configuration(changed)


def test_runtime_provenance_records_git_versions_and_platform(monkeypatch):
    def fake_git(arguments):
        values = {
            ('rev-parse', 'HEAD'): 'abc123',
            ('branch', '--show-current'): 'four-mode-asym-gridium-tests',
            ('status', '--porcelain=v1'): ' M local-file',
        }
        return values[tuple(arguments)]

    monkeypatch.setattr(runner, '_git_output', fake_git)
    provenance = runner.collect_runtime_provenance()
    assert provenance['git_revision'] == 'abc123'
    assert provenance['git_branch'] == 'four-mode-asym-gridium-tests'
    assert provenance['git_dirty'] is True
    assert provenance['git_status_porcelain'] == [' M local-file']
    for key in (
            'python_version', 'numpy_version', 'scipy_version',
            'qutip_version', 'platform'):
        assert provenance[key]


def test_publish_requires_valid_metadata_and_writes_marker_last(tmp_path):
    incomplete = tmp_path / '.result.incomplete-fixture'
    final = tmp_path / 'result'
    _write_complete_fixture(incomplete)
    assert not (incomplete / 'COMPLETE').exists()

    runner.publish_checkpoint_directory(incomplete, final)

    assert not incomplete.exists()
    assert (final / 'COMPLETE').is_file()
    metadata = runner.validate_complete_checkpoint_directory(final)
    assert metadata['configuration_sha256'] == runner.CONFIGURATION_SHA256


def test_publish_refuses_existing_final_without_moving_incomplete(tmp_path):
    incomplete = tmp_path / '.result.incomplete-fixture'
    final = tmp_path / 'result'
    _write_complete_fixture(incomplete)
    final.mkdir()
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        runner.publish_checkpoint_directory(incomplete, final)
    assert incomplete.is_dir()
    assert final.is_dir()


def test_incomplete_and_parameter_mismatched_outputs_are_rejected(tmp_path):
    incomplete = tmp_path / 'incomplete'
    _write_complete_fixture(incomplete)
    with pytest.raises(ValueError, match='COMPLETE missing'):
        runner.validate_complete_checkpoint_directory(incomplete)

    bad = tmp_path / 'bad'
    _write_complete_fixture(bad, fingerprint='wrong')
    final = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        runner.publish_checkpoint_directory(bad, final)
    assert bad.exists()
    assert not final.exists()


def test_interpolation_and_cross_grid_overlap_assignment():
    spatial3 = dict(n1max=0, N2=3, L2=1.0, N3=3, L3=1.0, N4=1)
    spatial5 = dict(n1max=0, N2=5, L2=1.0, N3=5, L3=1.0, N4=1)
    raw = np.zeros((9, 1, 3), dtype=complex)
    raw[0, 0, 0] = 1.0
    raw[4, 0, 1] = 1.0j
    raw[8, 0, 2] = -1.0

    interpolated = runner.interpolate_raw_states(raw, spatial3, spatial5)
    assert interpolated.shape == (25, 1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(interpolated.reshape(-1, 3), axis=0), 1.0,
    )

    permutation = [2, 0, 1]
    candidate = raw[..., permutation] * np.exp(0.37j)
    assignment, overlaps, matrix = runner.track_cross_grid_states(
        raw, candidate,
    )
    assert assignment.tolist() == [1, 2, 0]
    np.testing.assert_allclose(overlaps, 1.0)
    assert matrix.shape == (3, 3)


def test_checkpoint_hashes_detect_scientific_and_metadata_corruption(tmp_path):
    incomplete = tmp_path / '.result.incomplete-fixture'
    final = tmp_path / 'result'
    _write_complete_fixture(incomplete)
    runner.publish_checkpoint_directory(incomplete, final)

    (final / 'analysis.json').write_bytes(b'changed analysis')
    with pytest.raises(ValueError, match='file hash mismatch'):
        runner.validate_complete_checkpoint_directory(final)

    # Restore the scientific file, republish, then alter provenance metadata.
    second_incomplete = tmp_path / '.second.incomplete-fixture'
    second_final = tmp_path / 'second'
    _write_complete_fixture(second_incomplete)
    runner.publish_checkpoint_directory(second_incomplete, second_final)
    metadata_path = second_final / 'checkpoint.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['provenance'] = {'git_revision': 'tampered'}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='does not authenticate'):
        runner.validate_complete_checkpoint_directory(second_final)


def test_n51_loader_rejects_unpinned_bytes_before_using_metadata(tmp_path):
    for name in runner.N51_ORACLE_SHA256:
        (tmp_path / name).write_bytes(b'finite but not the verified oracle')
    with pytest.raises(ValueError, match='verified asymmetric oracle'):
        runner.load_verified_n51_checkpoint(tmp_path)


def _sample_analytic_states(spatial):
    x2 = np.linspace(-spatial['L2'], spatial['L2'], spatial['N2'])
    x3 = np.linspace(-spatial['L3'], spatial['L3'], spatial['N3'])
    theta3, theta2 = np.meshgrid(x3, x2, indexing='ij')
    fields = (
        np.exp(-((theta2 - 2.2) / 2.2) ** 2
               - ((theta3 + 4.1) / 3.0) ** 2)
        * np.exp(1j * (0.13 * theta2 - 0.07 * theta3)),
        (theta2 - 0.3)
        * np.exp(-((theta2 + 1.4) / 2.5) ** 2
                 - ((theta3 - 3.7) / 2.8) ** 2)
        * np.exp(-1j * 0.11 * theta3),
        (theta3 + 0.7)
        * np.exp(-((theta2 - 0.8) / 2.8) ** 2
                 - ((theta3 + 0.5) / 3.2) ** 2)
        * np.exp(1j * 0.09 * theta2),
    )
    n1_count = 2 * spatial['n1max'] + 1
    dimension = spatial['N3'] * spatial['N2'] * n1_count
    states = np.zeros((dimension, spatial['N4'], len(fields)), complex)
    quadrature = (x2[1] - x2[0]) * (x3[1] - x3[0])
    for state, field in enumerate(fields):
        tensor = np.zeros((
            spatial['N3'], spatial['N2'], n1_count, spatial['N4'],
        ), complex)
        tensor[:, :, state % n1_count, state % spatial['N4']] = (
            field * np.sqrt(quadrature)
        )
        vector = tensor.reshape(-1)
        states[..., state] = (
            vector / np.linalg.norm(vector)
        ).reshape(dimension, spatial['N4'])
    return states


def test_analytic_n51_to_n71_tracking_preserves_axes_order_and_phases():
    spatial51 = dict(
        n1max=1, N2=51, L2=11.0, N3=51, L3=14.0, N4=2,
    )
    spatial71 = dict(
        n1max=1, N2=71, L2=11.0, N3=71, L3=14.0, N4=2,
    )
    source = _sample_analytic_states(spatial51)
    expected = _sample_analytic_states(spatial71)
    mapped = runner.interpolate_raw_states(source, spatial51, spatial71)

    # The first state is anisotropic and off-center in both axes. These moments
    # expose a theta2/theta3 swap or either coordinate reversal.
    probability = np.abs(
        mapped[..., 0].reshape(71, 71, 3, 2)
    ) ** 2
    probability = probability.sum(axis=(2, 3))
    x2 = np.linspace(-11.0, 11.0, 71)
    x3 = np.linspace(-14.0, 14.0, 71)
    assert float(np.sum(probability * x2[None, :])) == pytest.approx(
        2.2, abs=2e-3,
    )
    assert float(np.sum(probability * x3[:, None])) == pytest.approx(
        -4.1, abs=2e-3,
    )

    permutation = [2, 0, 1]
    phases = np.exp(1j * np.array([0.4, -1.2, 2.1]))
    candidate = expected[..., permutation] * phases
    assignment, overlaps, _ = runner.track_cross_grid_states(
        mapped, candidate,
    )
    assert assignment.tolist() == [1, 2, 0]
    assert float(np.min(overlaps)) > 0.9999
    _, _, overlap_matrix = runner.track_cross_grid_states(mapped, candidate)
    confidence = runner.assignment_confidence_diagnostics(
        overlap_matrix, assignment,
    )
    assert all(row['state_specific_claim_safe'] for row in confidence)
    assert runner.ambiguous_subspace_clusters(
        overlap_matrix, assignment, confidence,
    ) == []


def _rotated_subspace(angle_degrees, leakage_degrees=0.0):
    angle = np.radians(angle_degrees)
    leakage = np.radians(leakage_degrees)
    rotation = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    reference = np.zeros((4, 1, 2), complex)
    reference[0, 0, 0] = 1.0
    reference[1, 0, 1] = 1.0
    orthogonal = np.zeros_like(reference)
    orthogonal[2, 0, 0] = 1.0
    orthogonal[3, 0, 1] = 1.0
    candidate = (
        np.cos(leakage)
        * np.einsum('dms,sq->dmq', reference, rotation, optimize=True)
        + np.sin(leakage)
        * np.einsum('dms,sq->dmq', orthogonal, rotation, optimize=True)
    )
    return reference, candidate


def test_near_degenerate_rotation_is_visible_as_assignment_ambiguity():
    reference, candidate = _rotated_subspace(45.0)

    assignment, overlaps, overlap_matrix = runner.track_cross_grid_states(
        reference, candidate,
    )
    np.testing.assert_allclose(overlaps, 1.0 / np.sqrt(2.0))
    # Individual labels are ambiguous, while the two-dimensional subspace is
    # exactly preserved. The full saved overlap matrix exposes this distinction.
    np.testing.assert_allclose(
        np.linalg.svd(overlap_matrix, compute_uv=False), 1.0,
    )
    confidence = runner.assignment_confidence_diagnostics(
        overlap_matrix, assignment,
    )
    assert all(row['competition_warning'] for row in confidence)
    assert not any(row['state_specific_claim_safe'] for row in confidence)
    clusters = runner.ambiguous_subspace_clusters(
        overlap_matrix, assignment, confidence,
    )
    assert len(clusters) == 1
    np.testing.assert_allclose(clusters[0]['singular_values'], 1.0)
    pathway = [{'path': [0, 1], 'score_product': 1.0}]
    annotated = runner.annotate_pathway_claims(
        pathway, confidence, clusters,
    )
    assert annotated[0]['claim_status'] == 'ambiguous'


def test_near_degenerate_assignment_becomes_unique_as_rotation_decreases():
    diagnostics = []
    for angle in (35.0, 10.0):
        reference, candidate = _rotated_subspace(angle)
        assignment, _, overlap = runner.track_cross_grid_states(
            reference, candidate,
        )
        confidence = runner.assignment_confidence_diagnostics(
            overlap, assignment,
        )
        subspace = runner.subspace_overlap_diagnostics(
            overlap, [0, 1], assignment[[0, 1]],
        )
        diagnostics.append((confidence, subspace))

    mixed, mixed_subspace = diagnostics[0]
    weak, weak_subspace = diagnostics[1]
    assert all(row['competition_warning'] for row in mixed)
    assert all(row['state_specific_claim_safe'] for row in weak)
    assert weak[0]['overlap_margin'] > mixed[0]['overlap_margin']
    np.testing.assert_allclose(mixed_subspace['singular_values'], 1.0)
    np.testing.assert_allclose(weak_subspace['singular_values'], 1.0)


def test_low_noncompetitive_overlap_defers_instead_of_claiming_ambiguity():
    overlap = np.diag([0.20, 1.0]).astype(complex)
    assignment = np.array([0, 1])
    confidence = runner.assignment_confidence_diagnostics(
        overlap, assignment,
    )
    clusters = runner.ambiguous_subspace_clusters(
        overlap, assignment, confidence,
    )
    assert clusters == []
    annotated = runner.annotate_pathway_claims(
        [{'path': [0, 1], 'score_product': 1.0}], confidence, clusters,
    )
    assert annotated[0]['claim_status'] == 'deferred'


def test_physically_changed_subspace_has_reduced_singular_values():
    reference, candidate = _rotated_subspace(45.0, leakage_degrees=30.0)
    assignment, _, overlap = runner.track_cross_grid_states(
        reference, candidate,
    )
    confidence = runner.assignment_confidence_diagnostics(
        overlap, assignment,
    )
    clusters = runner.ambiguous_subspace_clusters(
        overlap, assignment, confidence,
    )
    assert len(clusters) == 1
    np.testing.assert_allclose(
        clusters[0]['singular_values'], np.cos(np.radians(30.0)),
    )
    np.testing.assert_allclose(
        clusters[0]['principal_angles_degrees'], 30.0,
    )


def test_grid_phi_subspace_metrics_are_invariant_to_internal_rotation():
    matrix = np.array([
        [0.1, 0.2, 0.3j, -0.4, 0.5],
        [0.2, 0.6, -0.2j, 0.1, -0.3j],
        [-0.3j, 0.2j, 0.8, 0.7, 0.4j],
        [-0.4, 0.1, 0.7, -0.5, 0.6],
        [0.5, 0.3j, -0.4j, 0.6, 0.9],
    ], dtype=complex)
    angle = np.radians(37.0)
    unitary = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    transform = np.eye(5, dtype=complex)
    transform[np.ix_([2, 3], [2, 3])] = unitary
    rotated = transform.conj().T @ matrix @ transform
    clusters = [{'cluster_id': 0, 'reference_states': [2, 3]}]

    result = runner.grid_phi_subspace_diagnostics(
        matrix, rotated, clusters,
    )[0]
    np.testing.assert_allclose(
        result['reference']['logical_to_subspace_l2_norm']['0'],
        result['candidate']['logical_to_subspace_l2_norm']['0'],
    )
    np.testing.assert_allclose(
        result['reference']['logical_to_subspace_l2_norm']['1'],
        result['candidate']['logical_to_subspace_l2_norm']['1'],
    )
    np.testing.assert_allclose(
        result['reference']['internal_block_frobenius_norm'],
        result['candidate']['internal_block_frobenius_norm'],
    )
    np.testing.assert_allclose(
        result['reference']['internal_block_singular_values'],
        result['candidate']['internal_block_singular_values'],
    )
    np.testing.assert_allclose(
        result['reference']['subspace_to_complement_frobenius_norm'],
        result['candidate']['subspace_to_complement_frobenius_norm'],
    )
