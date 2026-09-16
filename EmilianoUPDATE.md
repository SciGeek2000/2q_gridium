## Work Block — 2026-09-07

Completed: Set up the Gridium research workflow, validated the IdealGridium environment, grounded the docs in the paper/Thomas notes, and audited the current Rabi_3Photon implementation.

Next: Confirm the remaining 3-photon control details with Thomas, then turn those answers into the first validated implementation task.

### Thomas feedback
-


## Work Block — 2026-09-08

Completed: Characterized the proposed three-photon pathways and protected-point selection rules, identifying important degeneracies and showing that several possible interpretations of the control scheme lead to substantially different implementations.

Next: Confirm Thomas's intended meaning of the three-photon process, pathway, tone structure, and drive operators before making further physics or implementation decisions.

### Thomas feedback
-


## Work Block — 2026-09-15

Completed: Built and validated the three-tone flux-drive control workflow and obtained a converged symmetric IdealGridium X180 candidate with 98.67% logical fidelity, 0.47% final leakage, ~0.50% peak leakage, and a 21.5 ns gate time.

Next: Use this validated symmetric X180 as the baseline for studying whether slight symmetry breaking can further reduce coherent gate error and leakage, then extend toward X90 and eventually the asymmetric four-mode Gridium model.

### Questions for Thomas
- Can you confirm that 0<->5, 5<->4, 4<->1 is the intended first three-flux-tone pathway?
- For the first symmetry-breaking study, which physical asymmetry should we vary first, and is there a preferred nominal magnitude/range?


## Work Block — 2026-09-16

Completed: Calibrated and validated a symmetric IdealGridium X90 candidate with 99.23% logical fidelity, 0.061% final leakage, and 0.198% peak leakage, added robust eigenstate tracking, and confirmed that a simple flux-bias offset away from the protected point does not provide a useful symmetry-breaking proxy.

Also began the multimode implementation path: documented the paper’s three-/four-mode and asymmetry hierarchy, implemented the source-faithful symmetric three-mode Eq. S23 model, and found that its fully extended representation does not converge cleanly because the common translated coordinate requires the compact-coordinate topology made explicit in Eq. S26. The next step is therefore to implement printed Eq. S26 as a separate compact-coordinate model before proceeding to the four-mode model.

### Questions for Thomas
- The supplement appears to contain a factor-of-two inconsistency between the inductive coefficient in Eqs. S23 and S26. Which coefficient should we treat as physically intended?
- For the eventual asymmetric four-mode model, should we derive the asymmetry directly from the full branch-level four-mode circuit, or is there an existing formulation you would prefer us to use?