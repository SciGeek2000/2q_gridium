# Open questions

## Scientific

- What exact parameter values and conventions define the protected operating point?
- What transition path and tone interpretation define the intended multiphoton / Raman-like bit flip?
- What target unitary, fidelity threshold, leakage threshold, and robustness criteria constitute complete single-qubit control?
- How does the planned four-mode asymmetric Gridium model relate to the current `ExpGridium` class, whose docstring describes a three-mode model?

## Architecture and reproducibility

- What is the canonical environment and command for running the full test suite reproducibly?
- When should machine-specific absolute paths in workflows and tests be replaced with repository-relative configuration?
- Which metadata—such as model parameters, pulse configuration, solver settings, and commit identifier—must accompany saved results?
- Should the `Rabi_3Photon` versus “two photon” naming discrepancy be resolved, and what scientific terminology should govern it?

## Confirmed current limitations

- `Simulations/Rabi_3Photon/workflow_funcs.py` contains incomplete paths and an inline bug note; no repair is documented here.
- `Tests/expgridium_test.py` and `Tests/cphase_test.py` are currently placeholders.
- Baseline pytest collection on 2026-09-07 did not complete in the inspected local interpreter because of existing dependency/import errors and an indentation error in `Tests/fluxonium_test.py`.

## Source / provenance

- Direct repository inspection on 2026-09-07.
- Scientific answers remain unrecorded until supported by an explicit project decision or authoritative source.
