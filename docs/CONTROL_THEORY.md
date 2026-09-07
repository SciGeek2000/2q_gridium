# Control theory

## Confirmed information

- `Simulations/Cphase/workflow_funcs.py` organizes configuration loading, coupled-system construction, solving, optimization helpers, and visualization around YAML inputs.
- `Simulations/Rabi_3Photon/main.ipynb` currently starts its demonstrated workflow with a soft `IdealGridium` instance and two pulse YAML files.
- `Simulations/Rabi_3Photon/workflow_funcs.py` is an unfinished single-qubit workflow using existing drive/evolution helpers from `Circuit_Objs/qchard_evolgates.py`.
- Existing code uses both `Rabi_3Photon` and “two photon” naming. This is a repository observation, not a settled physical interpretation.

## Needs verification

- The intended transition path, tone count, detunings, amplitudes, phases, and pulse timing.
- The target unitary and accepted fidelity and leakage metrics.
- Which parts of the CPhase workflow are structurally reusable without importing two-qubit scientific assumptions.

## Open questions

- Should the current “two photon” helper names describe the number of tones or the order of the intended process?
- What evidence will establish that the simulated operation is the intended multiphoton / Raman-like bit flip?

## Source / provenance

- `Simulations/Cphase/`
- `Simulations/Rabi_3Photon/`
- `Circuit_Objs/qchard_evolgates.py`
