# Project context

## Confirmed information

- Gridium is a superconducting-qubit research project using numerical simulation.
- The current research objective is to move from observing the qubit to complete single-qubit control in simulation before transferring that control to experiment.
- The immediate direction is a multiphoton / Raman-like bit-flip process near the protected operating point.
- Control work should begin with `IdealGridium`. Work on a more complicated four-mode asymmetric Gridium model comes later and requires explicit direction.
- Circuit implementations live in `Circuit_Objs/`, simulation workflows in `Simulations/`, tests in `Tests/`, and saved diagonalized objects in `etc/qubits/`.
- `Simulations/Cphase/` is the developed structural reference for the emerging `Simulations/Rabi_3Photon/` workflow.

## Needs verification

- Quantitative success criteria for “complete single-qubit control.”
- The exact scientific definition and parameterization of the protected operating point.
- The authoritative experimental and theoretical references for the planned control process.

## Open questions

- Which observables, fidelity measures, and leakage measures will define success for the single-qubit-control simulation?
- Which simulation artifacts should be retained as reproducible research results?

## Source / provenance

- Research direction: repository onboarding instructions supplied on 2026-09-07.
- Repository layout: direct inspection of `Circuit_Objs/`, `Simulations/`, `Tests/`, `Figures/`, `Notebooks/`, and `etc/`.
