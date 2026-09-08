# Project context

## Source boundary

Scientific context is drawn only from the [Gridium paper](<references/2509.14656v1 (3).pdf>) and the [Emiliano/Thomas running notes](<references/Emiliano and Thomas Gridium Running Notes.pdf>). Repository paths below are implementation observations, not additional physics sources.

## Confirmed scientific context

- Gridium is a superconducting artificial atom designed to host a doubly degenerate ($d=2$) grid-state computational subspace through the combination of coherent quantum phase slips and Cooper-quartet tunneling.
- The paper reports experimental spectra consistent with the predicted grid-like eigenstates, protected degeneracies, coherent dynamics on an allowed transition, and increased bit-flip resilience near symmetric bias. It does not demonstrate complete logical single-qubit control.
- The paper identifies a central tradeoff: approaching the ideal GKP regime suppresses charge and flux sensitivity, but also suppresses ordinary transition matrix elements and makes the computational states harder to observe and control directly.
- Higher excited levels remain important when low-order logical transitions are suppressed. In the paper's more-protected one-mode example, the `0-to-5` transition retains appreciable charge coupling across the examined flux range and supports dispersive characterization through virtual coupling.

## Confirmed project direction from the running notes

- The present goal is to move from observing Gridium to complete single-qubit control.
- The project intends to develop and simulate a complete single-qubit gate set before transferring it to experiment.
- The immediate proposed bit-flip direction is a three-photon Raman-like process near $\phi_{\mathrm{ext}}=0$ or $\pi$ with $\vartheta_{\mathrm{ext}}=\pi$, intended to produce an effective transition in the `0-to-1` subspace.
- Simulations should begin with `IdealGridium`. Only later should work move to the more complex four-mode asymmetric model described in the running notes.
- `Simulations/Cphase/` is a structural reference for the eventual `Rabi_3Photon` workflow, not a source of missing single-qubit physics.

## Repository map

- Circuit implementations live in `Circuit_Objs/`.
- Simulation workflows live in `Simulations/`.
- Tests live in `Tests/`.
- Research context and provenance live in `docs/`.

## What is not yet established

- A complete gate set has not been defined in the supplied sources.
- The three-photon pathway and control parameters are not specified.
- The sources do not define a target logical X unitary, including logical-basis phase conventions at exact degeneracy.
- No numerical fidelity, leakage, duration, robustness, or protection-preservation criteria are supplied for accepting a gate.
- The paper's discussion of the `0-to-5` transition concerns selection rules and readout; the running notes do not establish whether that level is part of the proposed bit flip.
- The relationship between the paper's four-mode parasitic-mode model and the running notes' planned "four mode asym gridium" is not fully specified.

## Source / provenance

- Gridium paper: main text pp. 1-6 and Supplementary Notes 1, 4, 5, and 11.
- Running notes: one-page "Overall Scope" and repository crash course.
- Repository locations: direct file-layout observation only.
