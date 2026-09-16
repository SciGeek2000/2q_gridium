# Results ledger

Use this ledger only for reproducible research results. Do not infer or backfill results from exploratory code, notebook output, or filenames.

## Confirmed information

- The portable Python 3.12 environment passed all six assertion-bearing `IdealGridium` baseline tests on 2026-09-07.
- The recorded eigenenergies below are direct outputs from the implementation at the stated revision; they are not an independent validation of the underlying physics.
- On 2026-09-15, the simultaneous three-tone flux-drive workflow produced a numerically converged X180-like gate candidate in symmetric soft `IdealGridium`.

## IdealGridium baseline: 2026-09-07

- **Code revision:** `5972b5e13bd389994317da38297e34819bbbc8a3`
- **Environment:** `environment-portable.yml`; Python 3.12.11, NumPy 2.2.6, SciPy 1.13.1, QuTiP 5.2.2, and scqubits 4.3.1.
- **Test result:** 6 passed, 2 deselected in 55.62 seconds; no pytest warnings or failures.
- **Environment provenance:** `environment.yml` is a machine-specific historical export. It was not used as the reproducible specification for this baseline.

Test command:

```shell
python -m pytest Tests/idealgridium_test.py -q \
  -k "phi_ext_spectrum or convergence or matrix_element"
```

Absolute eigenenergies returned by the default models, in their declared GHz units:

```text
soft IdealGridium (nlev_lc=230):
[-8.989491116435312, -8.989423459065708,
 -2.884198885966301, -2.8826394612845854,
 -2.3229700400060183, -2.321536612046256,
  2.9740802992688264, 3.0227468520908927]

hard IdealGridium (nlev_lc=600):
[-12.621640944306142, -12.621640944120946,
  -9.638569936534077, -9.638569931276143,
  -9.233079916781582, -9.23307991654523,
  -6.762227335723765, -6.76222726718135]
```

The public eigenbasis Hamiltonians reported Hermitian with zero residual. The maximum absolute element of `H - H.dag()` in the LC basis was `7.105427357601002e-15` for the soft model and `8.770761894538737e-15` for the hard model.

## Results

| Date | Goal | Model and configuration | Code revision | Procedure | Result and artifacts | Verification status | Source / provenance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-07 | Establish a local `IdealGridium` baseline | Default soft and hard `IdealGridium`; 8 returned levels; LC cutoffs 230 and 600 | `5972b5e13bd389994317da38297e34819bbbc8a3` | Targeted pytest command and direct spectrum/Hermiticity probe documented above | 6 passed, 2 deselected; spectra recorded above | Reproduced locally | Codex diagnostic run on macOS ARM64 using the pinned portable environment |
| 2026-09-15 | Establish a symmetric three-tone X180 control baseline | Soft `IdealGridium`; simultaneous flux drives on `0<->5`, `5<->4`, `4<->1`; calibrated 21.4963 ns cosine pulse | `e91216a` | Modest calibration search followed by solver, Hilbert-space, cutoff, coherence, and local-robustness validation | X180 fidelity `0.986696`; final leakage `0.004714`; peak leakage `0.004980` | Numerically converged and locally reproduced | `calibrate_simultaneous_x180.py`, `validate_simultaneous_x180.py`, experiment YAML |
| _YYYY-MM-DD_ | _What was tested_ | _Model, parameters, pulse, and solver settings_ | _Commit SHA_ | _Command or notebook_ | _Metrics and artifact paths_ | _Exploratory / reproduced / reviewed_ | _Author and supporting source_ |

## Symmetric IdealGridium three-tone X180 baseline: 2026-09-15

- **Code revision:** `e91216a`
- **Environment:** `environment-portable.yml`
- **Model:** soft `IdealGridium`
- **Parameters:** `E_L=0.5`, `E_C=0.5`, `E_s=4`, `E_2J=12`, `n_g=0`, `phi_ext=pi`
- **Candidate pathway:** `0<->5`, `5<->4`, `4<->1`
- **Control:** three simultaneous independent flux-drive tones with cosine envelopes
- **Gate duration:** `21.4963 ns`

### Bare transition frequencies

- `0<->5`: `6.667954504 GHz`
- `5<->4`: `0.001433428 GHz`
- `4<->1`: `6.666453419 GHz`

### Calibrated pulse parameters

- `A_05 = 0.407798`
- `A_54 = 0.303076`
- `A_41 = 0.396654`
- `phi_05 = 0`
- `phi_54 = +0.078660 rad`
- `phi_41 = -2.933753 rad`
- `Delta_05 = +0.000851913 GHz`
- `Delta_54 = +0.000996400 GHz`
- `Delta_41 = +0.000336940 GHz`

### Validated result

- **X180 logical gate fidelity:** `0.986696`
- **Final leakage:** `0.004714`
- **Peak sampled leakage:** `0.004980`
- **Gate duration:** `21.4963 ns`

The result remained stable under tighter solver tolerances, denser time sampling, retained-level expansion through 16 levels, and LC cutoff expansion through 460. Numerical changes in the reported gate metrics were below approximately `1.3e-5`.

Logical-superposition propagation confirmed coherent X-like action rather than only population transfer. The remaining error is predominantly a coherent rotation-axis/angle error.

### Leakage destination

Residual final leakage is almost entirely in levels 4 and 5:

- Average population in level 4: approximately `0.002316`
- Average population in level 5: approximately `0.002399`

Levels 2 and 3 contain less than approximately `1e-9`, and higher retained levels contain only numerical traces.

### Numerical convergence

Validation included:

- tighter and tighter solver tolerances;
- time sampling through 8 samples/ns;
- retained-level expansion through `nlev=16`;
- LC cutoff expansion through `N_LC=460`;
- propagation of `|0>`, `|1>`, `(|0>+|1>)/sqrt(2)`, and `(|0>+i|1>)/sqrt(2)`;
- local perturbations of amplitudes, phases, detunings, and gate duration.

The repository-default solver produced a unitarity residual of approximately `1.54e-2` for this gate calculation. Tight integration reduced this to approximately `7.9e-9` without materially changing the physical metrics.

Serious gate calculations should therefore use the validated tighter solver settings rather than the repository-default integration settings.

### Interpretation

The calibrated pulse is genuinely X-like rather than merely a population transfer. Both logical basis states approximately exchange, and logical superposition tests preserve coherence.

The remaining error is not dominated by leakage. Instead, the primary remaining imperfection is a coherent rotation-axis/angle error.

Local sensitivity tests indicate:

- the middle-tone amplitude and phase are the strongest controls of coherent gate fidelity;
- the `4<->1` carrier phase has the strongest observed association with leakage;
- small perturbations can improve the current metrics, so the frozen candidate is validated but is not necessarily a local optimum.

### Artifacts

- `Simulations/Rabi_3Photon/calibrate_simultaneous_x180.py`
- `Simulations/Rabi_3Photon/validate_simultaneous_x180.py`
- `Simulations/Rabi_3Photon/compare_x180_protocols.py`
- `Simulations/Rabi_3Photon/yamls/experiments/idealgridium_soft_candidate_x180.yaml`

### Verification status

Numerically converged and locally reproduced in the symmetric one-mode `IdealGridium` model.

The existing control test suite and the original `IdealGridium` regression baseline continued to pass after the control-work changes.

### Limitations

- The `0<->5<->4<->1` pathway still requires confirmation from Thomas as the intended first three-flux-tone pathway.
- No physical flux-line amplitude calibration has been established.
- No physical symmetry breaking has yet been introduced.
- No four-mode asymmetric Gridium model is implemented.
- X90 has not yet been calibrated.
- The current result applies only to the symmetric one-mode `IdealGridium` reference model.

## Needs verification

- Define the minimum metadata required before an entry may be treated as reproducible.
- Confirm the intended three-flux-tone pathway with Thomas.
- Establish the physical mapping between simulated flux-drive amplitude and the experimental control line.
- Determine the first physically meaningful asymmetry parameter and range for the next study.

## Open questions

- Where should large raw outputs and generated figures be stored and versioned?
- Which physical asymmetry should be introduced first when moving away from the perfectly symmetric model?
- What model-validation milestone should trigger the move from `IdealGridium` to the four-mode asymmetric Gridium implementation?

## Source / provenance

- Every result entry must identify its code revision, configuration, procedure, author, and verification status.
- The 2026-09-15 X180 result was generated from repository code at revision `e91216a` and validated numerically before being entered into this ledger.