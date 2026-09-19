## Work Block — 2026-09-07

Completed: Set up the Gridium research workflow and established the division between research planning/interpretation and repository implementation. Validated the IdealGridium Python/QuTiP environment and grounded the working notes in the Gridium paper and Thomas's guidance. Audited the existing `Rabi_3Photon` implementation and identified that the intended three-photon control mechanism, pathway, tone structure, and drive operator still needed to be pinned down before making substantive changes.

Next: Clarify the intended three-photon control scheme with Thomas, then turn those answers into the first bounded and reproducible control-validation task.

### Questions for Thomas

- Is the intended three-photon logical gate driven through flux modulation rather than charge modulation?
- Should the control machinery support three independently tunable drive frequencies from the beginning?
- Is there a preferred intermediate-state pathway that should be treated as the initial reference implementation?


## Work Block — 2026-09-08

Completed: Characterized the candidate three-photon pathways and the protected-point selection rules of the symmetric IdealGridium model. Identified important degeneracies and showed that several superficially similar interpretations of the control scheme lead to materially different implementations and matrix elements. This made it clear that the control physics should be resolved before optimizing a gate numerically.

Next: Use Thomas's clarification of the physical drive, pathway, and symmetry-breaking intent to define the first validated three-tone control experiment.

### Questions for Thomas

- Which physical flux coordinate/operator is intended to drive the logical gate?
- Is the first three-tone pathway intended to connect the logical states through the 0/5/4/1 manifold, or should a different intermediate manifold be used?
- Should perfect symmetry be treated only as a reference point, with slight fabrication asymmetry introduced intentionally for control?


## Work Block — 2026-09-15

Completed: Built and validated the three-tone flux-drive control workflow in symmetric IdealGridium and obtained a converged X180 candidate with 98.67% logical fidelity, 0.47% final leakage, approximately 0.50% peak leakage, and a 21.5 ns gate time. Tightened the time-domain solver enough to separate numerical error from coherent gate error and found that the remaining infidelity is primarily coherent axis/angle error rather than leakage. The result established a reliable symmetric baseline for testing X90 control and eventual symmetry breaking.

Next: Use the validated symmetric X180 as the control baseline, construct an X90 gate with the same methodology, and then determine how slight symmetry breaking affects fidelity, matrix elements, and leakage before moving to the full four-mode model.

### Questions for Thomas

- Can you confirm that `0 <-> 5`, `5 <-> 4`, `4 <-> 1` is the intended first three-flux-tone pathway?
- For the first symmetry-breaking study, which physical asymmetry should we vary first, and is there a preferred nominal magnitude or fabrication-relevant range?


## Work Block — 2026-09-16

Completed: Calibrated and validated a symmetric IdealGridium X90 candidate with 99.23% logical fidelity, 0.061% final leakage, and 0.198% peak leakage, and added robust eigenstate tracking for near-degenerate manifolds. Tested a simple external-flux offset as a symmetry-breaking proxy and found that it rotates/mixes the logical doublet rather than cleanly representing the fabrication asymmetry we want to study. Implemented and diagnosed the symmetric three-mode S23 model, finding that its fully extended representation has a common-coordinate/topology problem rather than a simple cutoff problem, then merged the validated work into `four-mode-asym-gridium-tests` as requested. Audited the shared four-mode code and identified `qchard_gridium_netlist.Gridium4Mode` as the authoritative candidate model; the initial bounded validation suite gave 13 passed and 5 intentionally skipped production-scale checks, while `N4=6 -> 8` was already stable and Stage-1 retained-state convergence remained the main numerical issue.

Next: Validate Stage-1 retention in the netlist-derived four-mode model before attempting spatial-basis convergence or porting the X90/X180 control workflow. Keep four-mode gate optimization on hold until the physical flux-drive operator is established.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat `qchard_gridium_netlist.Gridium4Mode` as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?


## Work Block — 2026-09-17

Completed: Continued validation of `Gridium4Mode` and isolated the Stage-1 shift-invert eigensolver as the dominant numerical bottleneck rather than Hamiltonian construction. Instrumentation showed that sparse LU factorization and fill-in dominated runtime; the default COLAMD ordering produced about 73.1x fill, while `MMD_AT_PLUS_A` reduced this to about 47.8x and substantially reduced memory and inverse-solve cost. Verified that the new ordering preserves the same Hamiltonian physics: at `k=100`, Stage-1 eigenvalues agreed to approximately `5.3e-13 GHz`, downstream energies agreed within approximately `0.000441 MHz`, and eigenpair/orthogonality residuals remained near numerical precision. A bounded `k=120` study still showed retained-space convergence failures, and a successful `k=140` solve was checkpointed for later state-overlap, operator, and cutoff-boundary analysis rather than rerunning expensive calculations.

Next: Resume only from the saved `k=140` artifacts, determine whether the retained-state error is ordinary slow Rayleigh-Ritz convergence or a more structured coupling problem, and use that evidence to decide whether a bounded `k=160` extension is scientifically justified.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat `qchard_gridium_netlist.Gridium4Mode` as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?


## Work Block — 2026-09-18

Completed: Extended the four-mode retained-state study through `k=180` and found that energy-ranked Stage-1 truncation can miss physically important higher-energy states: the `170 -> 180` step shifted `f06` by about 57.53 MHz even though the logical doublets remained well tracked. Post-processing traced the effect primarily to a small cluster around Stage-1 states 172/174/175, including a strong fourth-mode-assisted near resonance involving state 172, and showed that the dominant shell coupling is carried mainly through the `x2/x3` sector. A separate local Gridium sandbox then tested blind residual-, coupling-, channel-, QoI-, and operator-response enrichment methods using only the saved `k=180` parent data; the best blind constructions matched or beat the diagnostic hand basis spectrally at the same retained dimension, reaching about 2.81 MHz maximum first-six transition error and 0.363 MHz `f06` error, while operator-response/Krylov enrichment improved `d_theta` from roughly 20.6 MHz/rad for ordinary energy truncation to about 4.3 MHz/rad but still did not match the hand basis. Also productionized the validated optional `MMD_AT_PLUS_A` solver path and expanded the bounded physical-validation suite to 21 passed and 5 intentionally skipped tests covering flux periodicity, finite-cutoff charge-periodicity behavior, Hermiticity, and state-resolved `d_phi`/`d_theta` periodicity, with no physical inconsistency found.

Next: Stop brute-force energy-only `k > 180` growth and treat retained-subspace construction as the remaining numerical problem, especially blind representation of `d_theta` response. Once that issue is sufficiently controlled, resume `n1max/N2/N3` spatial-basis convergence and independent operator/model validation, then port the validated three-tone X90/X180 workflow to the asymmetric four-mode model.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat `qchard_gridium_netlist.Gridium4Mode` as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?
