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


## Work Block — 2026-09-25–2026-09-26

Completed: Incorporated Thomas's clarification that the current abstract four-mode control coordinate is `Gridium4Mode.grid_phi()`, with `phi_grid = theta3 + theta2/2`; `d_theta()` and `d_phi()` are external-flux response/sensitivity operators rather than the default microwave-drive proxy. Reproduced the frozen one-mode gates without reoptimization. X90 has duration `39.8007574 ns`, gate fidelity `0.9923100794`, process fidelity `0.9884651191`, final leakage `0.061239%`, and peak leakage `0.197513%`. X180 has duration `21.4963121 ns`, gate fidelity `0.9866983593`, process fidelity `0.9800475389`, final leakage `0.471441%`, and peak leakage `0.498001%`. Added permanent population, leakage, drive, intermediate-population, convergence, and CSV artifacts for these frozen validations.

Generalized the multitone workflow so `IdealGridium` uses `phi()` and `Gridium4Mode` uses `grid_phi()`, with every tone coupled through the same full operator and the existing per-target matrix-element normalization preserved. A reduced-basis four-mode three-tone smoke test demonstrated working `grid_phi` propagation; this was strictly a control-plumbing diagnostic, not a gate claim or convergence result.

Independent review caught two serious problems in the initial retained-basis analysis: a symmetric Sep-18 k180 checkpoint had accidentally been reused as an asymmetric N=51 reference, and noncontiguous shell-state overlap embedding was incorrect. The resulting numerical conclusions were withdrawn. State tracking was corrected so retained indices are embedded at their actual Stage-1 ranks and overlap assignments consistently reorder states, energies, transitions, tensors, and `grid_phi`.

Generated a fresh trustworthy asymmetric N=51 k180 oracle with `eps_J=0.10`, `eps_LK=0.05`, `n1max=4`, `N2=N3=51`, `L2=11`, `L3=14`, `N4=8`, Stage-1 `k=180`, `sigma=-24`, `which=LM`, `tol=1e-8`, and `MMD_AT_PLUS_A`. Its Stage-1 dimension is `23,409`, matrix nnz is `599,625`, LU nnz is `28,667,572`, fill ratio is `47.809x`, factorization time is `66.346 s`, ARPACK time is `170.118 s`, and the solve used `736` inverse solves. The maximum eigenpair residual is `5.433e-12`, median residual is `8.512e-13`, and orthogonality residual is `3.247e-12`.

The inherited symmetric shell-informed k170 basis failed for the asymmetric model, with about `109.9 MHz` maximum transition error, about `13.22%` error on the important `7->8` `grid_phi` edge, and material leakage-coupling errors. An asymmetry-specific shell audit identified Stage-1 state 170 as the dominant omitted contribution: removing it caused about `105.369 MHz` transition error and up to `0.144713` absolute primary-path matrix-element error. The selected asymmetric basis is `0..159 + {162,163,165,167,168,170,171,172,176,177}`. Against full asymmetric k180 at N=51, it gives `3.055 MHz` maximum transition error, `0.999938` minimum tracked overlap, `0.005236` absolute / `0.452%` maximum primary-path `grid_phi` error, and `0.009153` absolute / `2.13%` maximum audited logical/leakage `grid_phi` error. The leading pathway remains `0 -> 7 -> 8 -> 1`, with full-k180 primary edges `0->7 = 1.911695`, `7->8 = 1.157822`, and `8->1 = 1.815039`. This k170 basis is controlled for the next N=71 spatial-convergence calculation; spatial convergence itself is not established.

Solver/resource work confirmed `MMD_AT_PLUS_A` as the best tested SuperLU ordering. A local N=71 k180 solve was not practical on the current machine, and N=101 was rejected by resource preflight. Created and independently reviewed a portable external N=71 runner with explicit `--dry-run`/`--execute` modes, frozen physical/numerical settings, cryptographic pinning of the verified N=51 oracle, atomic incomplete/complete checkpoints, output hashing and checkpoint authentication, same-solve k170/k172/k175 analysis, overlap-based N=51-to-N=71 tracking, near-degenerate-subspace tracking, assignment-confidence diagnostics, SVD/principal-angle diagnostics, basis-invariant `grid_phi` subspace metrics, and `resolved`/`ambiguous`/`deferred` pathway claim gating. The runner is `READY FOR EXTERNAL N=71`. The final reviewed checkpoint is commit `f9fdd829d780d9d88f9919af4f2b3891e5bc5782` (`Validate asymmetric four-mode grid_phi workflow`) and is pushed to `upstream/four-mode-asym-gridium-tests`.

Next: Run the frozen asymmetric N=71 k180 reference on a higher-memory machine using commit `f9fdd82` and the reviewed runner; compare N=51 to N=71 with overlap/subspace tracking, validate the asymmetry-specific k170 basis at N=71, and only after spatial convergence is understood consider beginning four-mode X90/X180 optimization.

### Questions for Thomas

- Does the asymmetry-specific k170 basis and full-k180-reference strategy look reasonable for the N=71 spatial-convergence check?
- Is there a preferred higher-memory machine or cluster available for the N=71 run, or should we use any suitable 16+ GiB system?
