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

Completed: Calibrated and validated a symmetric IdealGridium X90 candidate with 99.23% logical fidelity, 0.061% final leakage, and 0.198% peak leakage; added robust eigenstate tracking; and confirmed that a simple flux-bias offset away from the protected point is not a useful symmetry-breaking proxy. Also implemented and diagnosed the symmetric three-mode S23 model, finding that its fully extended representation suffers from the common-coordinate topology problem rather than a simple cutoff issue.

Merged the validated control/multimode work into `four-mode-asym-gridium-tests` as requested and verified the merged branch with 34 control/multimode tests plus 6 IdealGridium regression tests passing. After the merge, audited the existing four-mode machinery and identified `qchard_gridium_netlist.Gridium4Mode` as the authoritative candidate model; its default validation suite gives 13 passed and 5 intentionally skipped production-scale convergence checks.

A bounded four-mode convergence diagnostic at the protected point showed that the explicit fourth-mode cutoff is already stable by `N4=6 -> 8`, while Stage-1 retained-state truncation is not yet converged: `nkeep=80 -> 100` still changes checked first-six transitions by as much as 17.96 MHz and several `d_phi` doublet-block singular values remain unstable, although `d_theta` is substantially better behaved. The original production `nkeep=320 -> 360` test was stopped after \~58 minutes without completing, so future validation will use bounded convergence ladders before retrying production-scale calculations.

Next: Extend only Stage-1 `nkeep` convergence at the fixed protected-point anchor (100, 120, 140, 160). Once Stage-1 retention is stable, proceed to spatial-basis convergence and independent flux/operator validation against the netlist/scqubits reference before adapting the multitone X90/X180 control workflow to the four-mode model. Do not optimize four-mode gates until the physical flux-drive operator is established.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat the branch-derived `qchard_gridium_netlist.Gridium4Mode` model as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?
