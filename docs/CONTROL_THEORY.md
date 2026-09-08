# Control theory

## Source boundary

Scientific statements here come only from the [Gridium paper](<references/2509.14656v1 (3).pdf>) and the [Emiliano/Thomas running notes](<references/Emiliano and Thomas Gridium Running Notes.pdf>). Existing workflow descriptions are labeled as repository observations.

## Confirmed control constraints from the Gridium paper

- Protection suppresses the charge and phase matrix elements that ordinary local drives use. The more closely the device approaches the ideal GKP regime, the less visible and directly addressable its low-order transitions become.
- More-protected spectra can still contain symmetry-allowed transitions involving higher excited levels and multiphoton processes. The paper demonstrates a Rabi chevron and an approximately 25 ns pi pulse on one such allowed transition, but it does not identify that transition as the proposed logical bit-flip process.
- In the more-protected one-mode example, the `0-to-5` transition is the only transition reported to have appreciable charge coupling throughout the examined flux range. Its virtual coupling to a capacitive resonator can produce a differential dispersive shift away from degeneracy, supporting characterization and readout with fast-flux assistance.
- The paper gives a separate example of higher-level surrogate control: swap `|0>` with `|3>` and `|1>` with `|4>`, operate nominally between `|3>` and `|4>`, then swap back. This is a possible control pattern, not the three-photon protocol described in the running notes.
- Operations that leave the protected manifold may lose intrinsic noise protection or cause leakage. The paper says protection-preserving gates remain an open challenge and mentions projective measurement, higher-level controlled-Z pathways, fast-flux pulses, Floquet driving, and erasure conversion as future directions rather than completed gate protocols.

(Gridium paper, main text pp. 5-6; Supplementary Note 1, PDF pp. 11-12.)

## Project control objective from the running notes

The project goal is complete single-qubit control: develop and simulate a complete single-qubit gate set before transferring it to the experimental platform. The immediate proposed bit-flip direction is a "3-photon Raman-like process" near

$$
\phi_{\mathrm{ext}}\in\{0,\pi\},\qquad \vartheta_{\mathrm{ext}}=\pi.
$$

The notes say that three simultaneous frequencies, later described as three different frequencies, should be combined with suitable amplitude to generate an effective transition within the qubit's `0-to-1` subspace. They do not specify a level pathway, drive Hamiltonian, intermediate states, frequencies, detunings, phases, pulse envelopes, or success metric. The wording therefore records a proposal, not a complete physical protocol.

The paper does not present or validate this particular three-photon Raman-like logical bit flip. It establishes the spectrum, selection-rule constraints, higher-level addressability, and general need for protected gate protocols that motivate investigating such a process.

## Repository implementation observations

- `Simulations/Cphase/` is the requested structural model for how `Simulations/Rabi_3Photon/` may eventually be organized; it is not a scientific source for the single-qubit protocol.
- The current Rabi workflow starts from `IdealGridium` and uses existing evolution helpers. No current function or filename should be treated as resolving the physics described above.
- The running notes point to the microwave evolution path and `H_drive_coeff_gate` as the anticipated implementation area, but no code change is authorized or implied by this document.

## Unresolved protocol definition

Before implementation, the project needs an explicit, reviewed control specification covering:

- the ordered transition pathway and coupling operator for each step;
- whether "three-photon" denotes perturbative order, three simultaneous tones, or three necessarily distinct carrier frequencies;
- real or virtual intermediate states and whether `|5>` is involved;
- carrier frequencies, amplitudes, relative phases, detunings, pulse shapes, and timing;
- the exact logical X target, including basis and phase conventions at degeneracy;
- the fidelity definition, leakage definition, thresholds, and robustness requirements.

## Source / provenance

- Gridium paper: main text "Temporal dynamics" and "Discussion and Outlook"; Supplementary Note 1.
- Running notes: "Overall Scope" and "Crash Course To 2Q-Gridium Repo."
