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