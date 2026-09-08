# Open questions

## Source boundary

These questions arise from gaps or ambiguities in the [Gridium paper](<references/2509.14656v1 (3).pdf>) and the [Emiliano/Thomas running notes](<references/Emiliano and Thomas Gridium Running Notes.pdf>). They are intentionally unresolved.

## Questions to ask Thomas before implementing the bit flip

1. What is the exact ordered state-to-state pathway for the proposed logical bit flip, including the initial state, final state, and every coupled transition?
2. What does "3-photon" mean here: a third-order process, three photons absorbed or emitted in a specified combination, three simultaneous drive tones, or three necessarily distinct carrier frequencies?
3. The notes say both "three frequencies" and "three different frequencies." Must all three carriers be distinct, and how is each frequency assigned to the pathway?
4. Which intermediate states are real populations and which are virtual states? What are their level indices in `IdealGridium`?
5. Is `|5>` an intermediate or endpoint in the proposed process? The paper establishes strong `0-to-5` charge coupling and a readout role, but does not connect it to this Raman-like bit flip.
6. Which physical operator does each tone couple through: charge, phase/flux, or a specified combination? How should the paper's selection rules constrain each leg?
7. What are the nominal carrier frequencies and the sign and definition of every single-photon and multiphoton detuning?
8. What amplitudes, relative phases, envelopes, start times, and durations are intended for the three tones? Does "combined in the right amplitude" mean one common amplitude or a required amplitude ratio?
9. Which protected point should be the default, $(\phi_{\mathrm{ext}},\vartheta_{\mathrm{ext}})=(0,\pi)$ or $(\pi,\pi)$? Must the operation remain there, or may it use a controlled flux excursion?
10. How should logical `|0>` and `|1>` be fixed and tracked at exact degeneracy, where an eigensolver may return an arbitrary basis within the degenerate subspace?
11. What is the exact target X gate, including allowed global phase, relative phase, basis convention, and required action on states outside the logical subspace?
12. Is the desired object a population swap, an $X_\pi$ rotation, or a full unitary suitable for composition with the rest of a gate set?
13. What fidelity metric should be used: state-transfer fidelity, average gate fidelity, process fidelity, or another measure? What numerical threshold constitutes success?
14. How is leakage defined, when is it evaluated, and what maximum leakage is acceptable? Must transient occupation of intermediate states also remain below a threshold?
15. What gate-duration and robustness requirements apply, including sensitivity to flux, charge, frequency, amplitude, and timing errors?
16. Must the gate preserve protection throughout, or is temporary departure from the protected manifold acceptable? If acceptable, how is the resulting error exposure assessed?
17. What evidence will distinguish the intended Raman-like mechanism from an off-resonant direct transition or another multiphoton process in simulation?
18. Which `IdealGridium` parameter set and Hilbert-space cutoff should be the first control baseline?

## Model-progression questions

- What validated milestone should trigger moving from `IdealGridium` to the three-mode model?
- Does the intended project sequence include an explicit three-mode control stage, or move directly from `IdealGridium` to a four-mode model?
- What exact Hamiltonian defines the planned "four mode asym gridium"? The paper gives four-mode parasitic-capacitance models and separate asymmetry corrections, but not one combined model under that name.
- Which high-frequency or array modes and which capacitance/asymmetry terms are required for the intended four-mode control simulation?
- How should logical states and control tones be tracked as the model changes and additional modes hybridize?

## Source ambiguities and non-establishments

- The running notes propose a three-frequency Raman-like bit flip but do not provide the transition diagram or mathematical drive model.
- The phrase "three different frequencies" suggests distinct simultaneous carriers, but the notes do not define "3-photon" precisely enough to settle that interpretation.
- The paper highlights `0-to-5` charge coupling for protected-state readout and dispersive response. It does not say that `|5>` participates in the proposed three-photon gate.
- The paper proposes a separate `0-to-3`, `1-to-4`, then `3-to-4` surrogate-state pattern. The sources do not equate it with the Raman-like proposal.
- The paper says protection-preserving gates are still required; the notes propose a process involving unspecified higher states. The sources do not establish whether that process preserves protection.
- The one-mode wavefunctions shown in main-text Fig. 1 are evaluated at $\phi_{\mathrm{ext}}=\pi/2$, whereas the identified pairwise-degenerate symmetric points are $0$ and $\pi$. These statements serve different purposes and should not be conflated.
- "Complete single-qubit control" and "complete gateset" are goals in the notes, but the required primitive gates and acceptance criteria are not defined.

## Architecture and reproducibility

- What is the canonical command for running the full repository test suite reproducibly?
- When should machine-specific absolute paths in workflows and tests be replaced with repository-relative configuration?
- Which metadata - model parameters, pulse definition, solver settings, environment, and commit identifier - must accompany control results?
- Should the `Rabi_3Photon` versus "two photon" repository naming discrepancy be resolved after the physical protocol is defined?

## Confirmed current limitations

- `Simulations/Rabi_3Photon/workflow_funcs.py` is unfinished; its existing names and partial implementation do not resolve the scientific questions above.
- `Tests/expgridium_test.py` and `Tests/cphase_test.py` are currently placeholders.
- The portable environment has been validated for the targeted `IdealGridium` baseline, not for the entire repository test suite.

## Source / provenance

- Gridium paper: main text pp. 2, 5-6; Supplementary Notes 1, 4, 5, and 11.
- Running notes: complete one-page document.
- Architecture limitations: direct repository observations, not scientific claims.
