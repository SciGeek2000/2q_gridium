# FourModeAsymGridium — validation & first results

Implementation of the exact four-mode asymmetric-KITE gridium Hamiltonian in
`Circuit_Objs/qchard_four_mode_asym_gridium.py`. All numbers below were produced with the
notebook kernel Python (3.9.6 / qutip 5.0.4). Parameters: SI Note 11 device fit, protection
bias (phi_ext, theta_ext) = (0, pi), ng = 0 unless noted.

## 1. Implementation exactness (code-level validation)

Two-stage hierarchical diagonalization vs brute-force full tensor-product diagonalization,
keeping the full stage-1 basis (tiny cutoffs: n_charge=2, Np=31, N3=4, N4=4):

| eps_J | eps_LK | ng | phi_ext | max |E_hier − E_brute| |
|---|---|---|---|---|
| 0 | 0 | 0.3 | 0.2 | 1.0e-12 GHz |
| 0.07 | 0.04 | 0.3 | 0.2 | 2.0e-12 GHz |

Pinned in `Tests/four_mode_asym_gridium_test.py` (3 passed). The hierarchy is an exact
unitary rearrangement of the four-mode model; truncation is the only approximation.

## 2. Convergence protocol (symmetric point)

nkeep_s1 is the controlling truncation (bases fixed at n_charge=8, Np=181, L=12, N3=10, N4=12):

| nkeep_s1 | delta_01 (MHz) | delta_23 | delta_45 |
|---|---|---|---|
| 40  | 362.1 | 47.9  | 429.4 |
| 80  | 320.7 | 53.7  | 363.2 |
| 160 | 294.6 | 85.3  | 321.8 |
| 320 | 288.5 | 108.5 | 289.4 |
| 640 | 288.5 | 107.3 | 288.3 |

Doubling every per-mode basis (n_charge 8->12, Np 181->241, N3 10->15, N4 8->12) moves
splittings by <2%. Class defaults set to the validated configuration (nkeep_s1=320).

Key solver notes (in the class):
- Stage 1 uses node-screened parameters (E_CS -> E_CS*eps_p/(E_CS+eps_p) = 0.89 GHz,
  E_LK -> E_LK*E_L/(E_LK+E_L) = 0.365 GHz); residuals restored exactly in stage 2.
  Without this exact regrouping the hierarchy converges chaotically (eps_p = 1 GHz makes
  the Born–Oppenheimer node elimination marginal).
- Shift-invert eigsh (sigma below the spectrum) replaces plain 'SA' Lanczos: ~10x faster.

## 3. Symmetric-limit spectrum (Step-0 physics verdict)

Converged, eps_J = eps_LK = 0:

- delta_01 = 288.5 MHz, delta_23 = 107.3 MHz, delta_45 = 288.3 MHz
- **The Step-0 expectation (i) — doublets degenerate to numerical precision — is NOT
  reproduced.** Plausible physics: the effective grid mode is soft (screened charging
  ~5 GHz; quartet potential emergent, not the 1-mode model's fundamental E_2J = 12 GHz),
  so tunneling splittings at the 10^2 MHz scale are natural.
- Outer-rung mismatch (E5−E0) − (E4−E1) = delta_01 + delta_45 = 578 MHz already at eps = 0:
  tone A cannot be simultaneously resonant with both outer rungs; tone B sits at
  |delta_01| ≈ 288.5 MHz (a genuine RF frequency) regardless of asymmetry.
- ng dependence of delta_01: flat to ~1.5% over ng in [0, 0.5] (charge protection OK).
- Selection rules at eps=0: <0|D_phi|1> ~ 1e-14 and <0|n_1|1> ~ 1e-14 (exactly dark,
  checkerboard OK); <0|D_theta|1> ~ 2e-2 (NOT dark — deviates from Step-0 (iii)).

## 4. Asymmetry sweep (Step-1 preview; converged settings)

| eps_J | eps_LK | d01 (MHz) | d23 | d45 | \|<4\|Dth\|5>\| | \|<0\|Dth\|5>\| | \|<1\|Dth\|4>\| | \|<0\|Dth\|1>\| |
|---|---|---|---|---|---|---|---|---|
| 0    | 0    | 288.5 | 108.5 | 289.4 | 0.098 | 0.015 | 0.021 | 0.019 |
| 0.03 | 0    | 288.5 | 108.3 | 289.1 | 0.100 | 0.015 | 0.021 | 0.019 |
| 0.10 | 0    | 288.4 | 106.7 | 286.3 | 0.120 | 0.018 | 0.025 | 0.021 |
| 0.03 | 0.03 | 288.2 | 108.3 | 289.0 | 0.102 | 0.015 | 0.022 | 0.019 |

**Multimode suppression of the asymmetry mechanism is near-total**: even eps_J = 10%
changes splittings by ~1% and brightens the middle rung by ~23% (vs the 1-mode caricature's
~10x brightening at 3%). This quantitatively reproduces the paper's multimode finding of
"no appreciable spectral change up to 10%" — pinning this suppression factor was the stated
goal of Step 1. Inductor asymmetry (eps_LK) adds nothing beyond eps_J.

## 5. Step-0 production run (Simulations/four_mode_gkp/step_0.ipynb)

Full checklist executed at production cutoffs (~3.7 h). Headline additions:

- **Re-centered degeneracy point**: the theta_ext sweep shows delta_01 closing linearly
  (V-shaped level crossing) at **theta* ~ 3.276 = pi + 0.134**, where delta_01 = 7.8 MHz on
  the refine grid (vs 288.5 MHz at pi). The Step-0 degeneracy expectation (i) is satisfied
  at the re-centered bias; (0, pi) is simply not this Hamiltonian's degeneracy point
  (theta -> 2pi - theta is not a symmetry: the rhombus flux enters as theta/2). At theta*:
  delta_23 = 400 MHz, delta_45 = 247 MHz, outer-rung mismatch 255 MHz, ladder rungs ~20x
  brighter (|<0|D_theta|5>| = 0.30).
- **ng check (iv)**: spectrum periodic under ng -> ng+1 to 1e-7 GHz; delta_01 charge
  dispersion 0.069 MHz over the full period.
- **Basis doubling at eps_J = 0, 3%, 15%**: charge/grid/range knobs < 0.1%; delta_23 the
  softest observable (2.4% under nlev_node, 1.1% under nkeep); tiny matrix elements
  (|D05| ~ 0.015) swing ~175% when nlev_delta doubles -> use nlev_delta >= 20 for
  matrix-element maps.
- **Node-stiffness proxy (ii)**: delta_01(eps_p -> 30 GHz) = 214 MHz at (0, pi) — the finite
  splitting is not a node-truncation artifact.
- **Selection rules (iii)**: D_phi and n_1 exactly dark (1e-14) at both biases; D_theta not
  dark, as required by the pair's linear theta-dispersion (D_theta = dH/d theta_ext).
- phi_ext sweep exactly mirror-symmetric about 0 (sweet spot along phi).

## 5b. Open items before building Step-1 maps

1. **Level-pairing check**: delta_01 ≈ delta_45 (288.5 vs 288.3 MHz) is suspicious —
   higher doublets normally split more than lower ones. A state-character analysis
   (per-eigenstate <m1>, grid-mode spread, Delta/node occupation) should confirm that
   (0,1)/(4,5) really are the gate ladder's doublets before trusting the tone placement.
2. **Spectrum-vs-flux vs the paper** (Fig. 3f / ED Fig. 3b): sweeps are now computed in
   step_0.ipynb; the manual comparison remains — it also adjudicates whether the theta*
   offset is physics or a flux-convention difference. The R_x working point must be
   re-derived at theta* (delta_01 -> 0 there changes the tone placement qualitatively).
3. Four-mode vs three-mode cross-check at small eps (Step-0 (ii)) — needs the 3-mode
   model implemented.
4. Adjudicate the two Step-0 mismatches (finite splittings; D_theta not dark) with the
   design author.
