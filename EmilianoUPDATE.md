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


## Work Block — 2026-09-17

Completed: Continued validation of the netlist-derived `Gridium4Mode` model and isolated the Stage-1 eigensolver as the main numerical bottleneck. With the original shift-invert configuration, `k=120` exceeded 300 s and `k=160` exceeded 600 s without returning eigenpairs, while Hamiltonian assembly itself was negligible.

Instrumented the `k=100` Stage-1 solve and found that sparse LU factorization dominated runtime and produced substantial fill-in: the default COLAMD factorization required about 73.6 s, produced roughly 43.8 million LU nonzeros (73.1x fill), and pushed peak memory near 1.9 GiB. Benchmarking SuperLU orderings identified `MMD_AT_PLUS_A` as the best alternative, reducing LU fill to 47.8x, materially lowering memory use and inverse-solve cost while preserving the identical Hamiltonian.

Validated `MMD_AT_PLUS_A` at `k=100`: Stage-1 eigenvalues agreed with the original solver to within approximately `5.3e-13 GHz`, downstream four-mode energies agreed within `0.000441 MHz`, and eigenpair/orthonormality residuals remained near numerical precision. Using the validated ordering, a single `k=120` solve completed successfully in about 33.9 s (`~66.6 s` total diagnostic runtime), allowing the same eigensystem to be reused for `nkeep=100,110,120`.

The `k=120` convergence study showed that Stage-1 retention is still not fully converged. At `110 -> 120`, `f02`, `f03`, and `f06` remained above the spectral thresholds; five `d_phi` singular values and one `d_theta` singular value also failed. A follow-up audit found that the behavior is best explained by genuine nested Rayleigh-Ritz truncation error and differential virtual-state dressing, rather than eigensolver noise or arbitrary rotations within the near-degenerate doublets. Energy-only Stage-1 truncation remains mathematically sound but converges slowly.

A bounded `k=140` Stage-1 solve was then completed successfully using `MMD_AT_PLUS_A`, and all downstream truncations for `nkeep=100,110,120,130,140` were completed. The full Stage-1 eigenvectors, projected operators, and downstream eigensystems were saved under `/private/tmp` before the usage limit was reached. The expensive computation is therefore preserved, but the final `k=140` convergence, overlap/state-tracking, and cutoff-boundary diagnostics have not yet been evaluated.

Next: Resume from the saved `k=140` artifacts only; do not rerun the Stage-1 solve. Complete the `120 -> 130 -> 140` convergence tables, low-energy state/subspace overlaps, and Stage-1 cutoff-boundary weight diagnostics. Use those results to decide whether Stage-1 retention is sufficiently converged for `n1max/N2/N3` spatial-basis checks or whether a `k=160` calculation is scientifically justified.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat the branch-derived `qchard_gridium_netlist.Gridium4Mode` model as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?


## Work Block — 2026-09-18

Completed: Extended the four-mode Stage-1 retained-state convergence study through `k=180` using the validated `MMD_AT_PLUS_A` shift-invert solver. The `k=160` calculation showed near-convergence, but the subsequent `160 -> 170 -> 180` ladder revealed a large structured change when Stage-1 states 170-179 entered: `f06` shifted by 57.53 MHz at `170 -> 180`, several `d_phi` blocks regressed, and one `d_theta` block failed, while the logical doublets and low-energy subspaces remained very well tracked. This ruled out simple state-labeling or gauge artifacts and showed that pure energy-ranked Stage-1 truncation can miss important states even when direct cutoff-boundary weight is small.

Post-processing of the saved `k=180` eigensystem identified the effect as a small-cluster coupling problem rather than broad uniform truncation error. Stage-1 states 172, 174, and 175 dominate the newly admitted shell; state 172 participates in a particularly strong fourth-mode-assisted near resonance with only about 5.85 MHz detuning and roughly 92.7 MHz coupling. The dominant Stage-2 shell coupling is carried primarily by the `x2/x3` quadrature, which also explains why `d_phi` is considerably more sensitive than `d_theta`.

A same-dimension proof-of-concept selected-basis benchmark was then performed entirely from the saved `k=180` data, with no new Stage-1 eigensolve. Ordinary energy-only `nkeep=170` was compared against a 170-state basis containing states `0-159` plus coupling-important states `165,169,171,172,173,174,175,176,178,179`. Relative to the full `nkeep=180` reference, the selected basis reduced the maximum low-energy error by about 29.5x, the `f06` error by about 17.7x, the maximum `d_phi` singular-value error by about 12.2x, the maximum `d_theta` singular-value error by about 12.7x, and the largest omitted-space residual by about 5.3x. Some individual transitions, notably `f05`, became worse, so this is a diagnostic proof of concept rather than a production selection algorithm.

Conclusion: increasing `nkeep` by energy ordering alone is no longer the preferred numerical strategy. The next step is to develop and benchmark a deterministic residual/coupling-informed Stage-1 state-selection rule that can identify important states without first knowing the full high-`k` solution. No further `k>180` solve should be run until that strategy is tested.

### Questions for Thomas

- For the physical flux drive, is the intended control line primarily modulation of `phi_ext`, `theta_ext`, or a calibrated linear combination of the two?
- Can we treat the branch-derived `qchard_gridium_netlist.Gridium4Mode` model as the intended physical four-mode asymmetric model going forward?
- Is there an author-approved resolution of the factor-of-two discrepancy in the inductive term between Eqs. S23 and S26?
