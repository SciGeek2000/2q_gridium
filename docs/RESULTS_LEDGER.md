# Results ledger

Use this ledger only for reproducible research results. Do not infer or backfill results from exploratory code, notebook output, or filenames.

## Confirmed information

- The portable Python 3.12 environment passed all six assertion-bearing `IdealGridium` baseline tests on 2026-09-07.
- The recorded eigenenergies below are direct outputs from the implementation at the stated revision; they are not an independent validation of the underlying physics.

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
| _YYYY-MM-DD_ | _What was tested_ | _Model, parameters, pulse, and solver settings_ | _Commit SHA_ | _Command or notebook_ | _Metrics and artifact paths_ | _Exploratory / reproduced / reviewed_ | _Author and supporting source_ |

## Needs verification

- Define the minimum metadata required before an entry may be treated as reproducible.

## Open questions

- Where should large raw outputs and generated figures be stored and versioned?

## Source / provenance

- Every result entry must identify its code revision, configuration, procedure, author, and verification status.
