# Multimode Gridium implementation specification

## Scope and source boundary

This document separates equations stated by the Gridium paper from repository
observations and from modeling choices that the sources do not settle. Its
primary scientific source is the Gridium paper and Supplementary Information,
especially Supplementary Notes 4, 5, and 11. Repository statements refer to
the source tree at the start of the 2026-09-16 work block.

The paper does **not** state one combined Hamiltonian containing the fourth
parasitic mode, cross-KITE capacitance, Josephson asymmetry, and inductive
asymmetry. Such a model must not be assembled silently from separate formulas.

## 1. Symmetric three-mode Hamiltonian

### Paper equation S23

After eliminating the high-energy variable associated with vanishingly small
cross-circuit capacitance using a Born--Oppenheimer approximation, the paper
gives

\[
\begin{aligned}
\hat H_{3}={}&2E_C(\hat n_\Sigma^2+\hat n_\Delta^2)
 +4E_{CS}\hat n_S^2\\
&-2E_J\cos\hat\phi_\Sigma\cos\hat\phi_\Delta
 -E_{JS}\cos(\hat\phi_S-\phi_{\rm ext})\\
&+E_L'(\hat\phi_\Sigma-\hat\phi_S)^2
 +E_{LK}(\hat\phi_\Delta-\vartheta_{\rm ext}/2)^2,
\end{aligned}
\tag{S23}
\]

with

\[
E_L'=\frac{2E_{LK}E_L}{2E_{LK}+E_L}.
\]

The signs and factors above are the S23 conventions. In particular, both
charging terms are positive, the Josephson terms are negative, and the
inductive terms are positive.

The variables originate from the branch phases in Supplementary Note 4:

\[
\phi_\Sigma=(\varphi_1+\varphi_2)/2,\qquad
\phi_\Delta=(\varphi_1-\varphi_2)/2,
\]

with \(\phi_S\) the S-junction branch phase. Their canonical conjugates are
\(n_\Sigma\), \(n_\Delta\), and \(n_S\), respectively. Canonical
quantization uses the usual phase/number pairs; the paper writes the energy
Hamiltonian directly and quotes energies in \(h\,\mathrm{GHz}\).

Equations S24--S25 define the derived coordinates

\[
\phi_1=\phi_S,\qquad
\phi_2=\phi_S-\phi_\Sigma,\qquad
\phi_3=\phi_\Delta,
\tag{S24}
\]

and

\[
n_1=n_S+n_\Sigma,\qquad
n_2=-n_\Sigma,\qquad
n_3=n_\Delta.
\tag{S25}
\]

### External-flux conventions

- \(\phi_{\rm ext}\) appears only in the S-junction cosine as
  \(-E_{JS}\cos(\phi_S-\phi_{\rm ext})\) in S23.
- \(\vartheta_{\rm ext}\) appears as the half-flux displacement
  \(E_{LK}(\phi_\Delta-\vartheta_{\rm ext}/2)^2\).
- The paper's multimode spectra include scans at
  \(\vartheta_{\rm ext}=0,\pi\), scans at
  \(\phi_{\rm ext}=0,\pi\), and protected-point calculations at
  \((\phi_{\rm ext},\vartheta_{\rm ext})=(0,\pi)\). These uses do not imply
  an additional periodic symmetry of the extended quadratic coordinate.
- The variables are dimensionless reduced phases. A dynamical coupling in
  physical flux units additionally requires the appropriate flux conversion
  and line calibration, neither of which is specified by S23.

The paper-defined bias derivatives of S23 are therefore

\[
\frac{\partial H_3}{\partial\phi_{\rm ext}}
=-E_{JS}\sin(\phi_S-\phi_{\rm ext}),
\qquad
\frac{\partial H_3}{\partial\vartheta_{\rm ext}}
=-E_{LK}(\phi_\Delta-\vartheta_{\rm ext}/2).
\]

They are derivatives with respect to dimensionless bias coordinates. Without
a circuit-to-line mutual-inductance map, neither should be labeled *the*
physical drive operator.

### Parameter provenance

The circuit energies are traced to Supplementary Note 4:

- \(E_C=e^2/(2C_J)\), for the symmetric KITE junction capacitance.
- \(E_{CS}=e^2/(2C_{JS})\), for the S-junction capacitance.
- \(E_J\) is the mean KITE Josephson energy in the symmetric model.
- \(E_{JS}\) is the S-junction Josephson energy.
- \(E_{LK}\) is the KITE inductive energy.
- \(E_L\) is the S-branch inductive energy.
- \(E_L'\) is the reduced inductive coefficient defined immediately below
  S23, not an independent fit parameter.

Table S1 supplies four S23 parameter regimes, in \(h\,\mathrm{GHz}\):

| Regime | \(E_J\) | \(E_C\) | \(E_L\) | \(E_{LK}\) | \(E_{JS}\) | \(E_{CS}\) |
|---|---:|---:|---:|---:|---:|---:|
| a | 5 | 0.5 | 1 | 1 | 4 | 8 |
| b | 10 | 0.5 | 1 | 1 | 4 | 8 |
| c | 10 | 0.5 | 0.5 | 0.5 | 4 | 8 |
| d | 10 | 0.5 | 0.2 | 0.2 | 4 | 8 |

The corresponding effective one-mode parameters printed on the right side of
Table S1 are least-squares fit results. They are not parameters of S23.

## 2. S21 and S26 source inconsistencies

These discrepancies must remain visible until resolved by an author or a
corrected source.

### S21 inconsistency

The printed high-energy Hamiltonian in S21 does not follow consistently from
S20: it uses an \(E_{LS}\)-like symbol that is not defined in the surrounding
circuit definitions and pairs the high-energy coordinate with operators that
do not match the equilibrium position printed in S22. In contrast, S22 gives

\[
\langle\varphi\rangle=
\frac{2E_{LK}\phi_\Sigma+E_L\phi_S}{2E_{LK}+E_L},
\tag{S22}
\]

which is consistent with minimizing the \(\varphi\)-dependent terms in S20.
S22 and the explicit S20 potential are therefore the safer provenance for the
Born--Oppenheimer reduction. The repository must not encode the literal S21
print as an independently trusted physical equation.

### S23/S26 factor-of-two discrepancy

Substituting S24 into the S23 inductive term gives

\[
E_L'(\phi_\Sigma-\phi_S)^2
=\frac{2E_{LK}E_L}{2E_{LK}+E_L}\phi_2^2.
\]

The printed S26 instead assigns \(\phi_2^2\) the coefficient

\[
\frac{E_{LK}E_L}{2E_{LK}+E_L},
\]

which differs by a factor of two. A coordinate substitution alone cannot
produce that change. Stage A therefore implements S23 directly and does not
implement or “correct” S26.

S26 also introduces the compact-coordinate representation and offset charge,

\[
4E_{CS}(n_1+n_2+n_g)^2+2E_C(n_2^2+n_3^2),
\]

with the transformed Josephson and inductive terms. That representation needs
a mixed compact/extended numerical basis and is outside Stage A.

## 3. Four-mode extension and parasitic capacitances

### Cross-circuit capacitance \(C_p\): S20

Before the Born--Oppenheimer elimination, S20 retains the fourth conjugate
pair \((\varphi,n)\):

\[
\begin{aligned}
H_4={}&2E_C(n_\Sigma^2+n_\Delta^2)+4E_{CS}n_S^2+4\epsilon_p n^2\\
&-2E_J\cos\phi_\Sigma\cos\phi_\Delta-E_{JS}\cos\phi_S\\
&+E_{LK}\left[(\phi_\Sigma-\varphi)^2+
(\phi_\Delta-\vartheta_{\rm ext}/2)^2\right]\\
&+E_L(\phi_S-\varphi+\phi_{\rm ext})^2,
\end{aligned}
\tag{S20}
\]

where \(\epsilon_p=e^2/(2C_p)\). Thus finite \(C_p\) creates a dynamical
fourth mode; \(\epsilon_p\) is an energy-like charging coefficient, not a
capacitance. Note also that the external-flux gauge in S20 is not written in
the same placement as in reduced S23. A four-mode implementation must preserve
the equation's gauge consistently rather than copying individual S23 terms.

### Cross-KITE capacitance \(C_K\): S29--S30

The paper introduces \(C_K\) between the outer conducting pads. It changes the
coefficient of the differential-mode kinetic term. S30 has the S20 structure,
but replaces \(2E_C n_\Delta^2\) by \(2E_C'n_\Delta^2\), where

\[
E_C'=\frac{e^2}{2(C_J+C_K/2)}
=\frac{2E_C\epsilon_K}{E_C+2\epsilon_K},
\qquad \epsilon_K=\frac{e^2}{2C_K}.
\tag{S30 definition}
\]

Therefore:

- \(C_p\) creates the fourth kinetic degree of freedom through
  \(4\epsilon_p n^2\).
- \(C_K\) does not create another mode in S30; it renormalizes the
  \(n_\Delta^2\) coefficient.
- \(C_p,C_K\) are physical capacitances, whereas
  \(\epsilon_p,\epsilon_K\) and \(E_C'\) are energies.

## 4. Asymmetry model

Supplementary Note 5 defines both corrections explicitly relative to the
original symmetric **three-mode Hamiltonian S23**, writing
\(H=H_0+H_\epsilon\).

For KITE-junction asymmetry,

\[
E_{J1,2}=E_J^{\rm mean}(1\pm\epsilon_J),
\qquad
H_{\epsilon_J}=2\epsilon_JE_J
\sin\phi_\Sigma\sin\phi_\Delta.
\tag{S31}
\]

For KITE-inductor asymmetry,

\[
E_{LK1,2}=E_{LK}^{\rm mean}(1\pm\epsilon_{LK}),
\]

and the paper's first-order correction is

\[
H_{\epsilon_{LK}}=
2\epsilon_{LK}\frac{E_LE_{LK}}{E_L+E_{LK}}
\phi_\Delta(\phi_S-\phi_\Sigma).
\tag{S32}
\]

These are perturbative corrections to S23. The paper notes that sufficiently
large inductive asymmetry can hybridize the slow differential mode with the
fast S mode and invalidate the Born--Oppenheimer approximation. S31--S32 do
not by themselves authorize adding the same terms to S20 or S30.

## 5. Combined four-mode asymmetric model: unresolved

The paper does not explicitly define a Hamiltonian that simultaneously
contains:

1. the S20 fourth mode from finite \(C_p\),
2. the S30 \(C_K\) correction,
3. S31 Josephson asymmetry, and
4. S32 inductive asymmetry.

Combining them may be derivable from the unreduced asymmetric circuit
Lagrangian, but that derivation is not printed. It is therefore an unresolved
modeling task, not a source-grounded equation ready for transcription.

## 6. Repository audit

### Existing code

- `Circuit_Objs/qchard_idealgridium.py` is the validated one-mode effective
  model. It is not a numerical base class for S23 and must remain unchanged.
- `Circuit_Objs/qchard_expgridium.py` is an incomplete historical stub. Its
  Hamiltonian and operator methods contain tuple-producing trailing commas,
  recursive accessors, and undefined attributes. Its parameter dictionaries
  resemble Table S1, but the class is not a working implementation of S23,
  S26, S20, or S30.
- The exploratory three-mode notebook does not supply an executable,
  independently validated multimode Hamiltonian. It is useful only as
  historical exploration.
- Existing qchard objects establish useful public conventions: `levels`,
  `eigvecs`, `H`, `freq`, operator accessors, matrix elements, energy units in
  GHz, and cache invalidation when parameters change.
- `Circuit_Objs/qchard_statetracking.py` can compare eigensystems represented
  in the same Hilbert space. Different tensor cutoffs need a physical embedding
  before overlaps are meaningful.
- `Circuit_Objs/qchard_evolgates.py` establishes the repository convention of
  multiplying a GHz Hamiltonian by \(2\pi\) for evolution with time in ns.

### Mapping to S23

| Required object | Source expression | Repository status before Stage A |
|---|---|---|
| \(\phi_\Sigma,n_\Sigma\) | S19/S23 canonical pair | absent |
| \(\phi_\Delta,n_\Delta\) | S19/S23 canonical pair | absent |
| \(\phi_S,n_S\) | S19/S23 canonical pair | absent |
| tensor products | three independent modes | generic QuTiP infrastructure reusable |
| cosine operators | S23 | generic matrix-exponential methods reusable |
| \(\phi_{1,2,3},n_{1,2,3}\) | S24--S25 | absent |
| sparse low-energy solve | lowest S23 eigenpairs | SciPy/QuTiP infrastructure reusable |
| state tracking | overlap and phase alignment | reusable only at equal tensor dimensions/bases |
| flux-line operator | circuit-specific coupling | not uniquely specified |

The historical `ExpGridium` class should be replaced by a new clean object for
each source-grounded model rather than patched until its intended equation is
ambiguous.

## 7. Safest staged architecture

### Stage A: symmetric three-mode S23

Implement S23 literally in an extended phase basis, with independent
\(\Sigma,\Delta,S\) truncations, sparse tensor operators, named primitive and
derived coordinates, bias derivatives, and a sparse low-energy eigensolver.
Do not add compact-coordinate or S26 behavior.

### Stage B: validate S23

Establish convergence by mode, reproduce Table S1 qualitative spectra and
selection-rule plots, and compare against any author-provided numerical data.
Resolve whether the oscillator basis is efficient enough for the protected
regimes. Do not call the model quantitatively validated merely because it is
runnable.

### Stage C: symmetric four-mode S20/S30

Implement the fourth \((\varphi,n)\) pair from the complete printed equation.
Treat the S30 cross-KITE correction as a separately selectable, sourced
parameterization. Preserve S20/S30's flux gauge. Do not add S31/S32 yet.

### Stage D: validate the symmetric four-mode limit

Check recovery of S23 as \(\epsilon_p\) becomes the largest scale, recovery of
S20 as \(C_K\to0\) (equivalently \(E_C'\to E_C\)), mode convergence, and the
paper's stated low-energy agreement and higher-mode behavior.

### Stage E: derive and add asymmetry

First reproduce S31 and S32 on S23. A combined four-mode asymmetric model
requires either a fresh circuit derivation reviewed in the repository or
author confirmation; it must not be created by unqualified term addition.

### Stage F: state tracking and control integration

Track logical and relevant excited states within a fixed tensor basis, add
correct embeddings for cutoff changes, and only then map physical control
ports to operators. Integrate multitone evolution after the drive coupling and
units are documented.

## 8. Validation requirements

### Stage A/B

- Hermiticity of every primitive operator, derived coordinate, derivative,
  and full Hamiltonian.
- Exact implementation of \(E_L'\), S23 bias shifts, and S24--S25 identities.
- Correct tensor dimension and mode order.
- Independent convergence in \(N_\Sigma,N_\Delta,N_S\), including at least the
  lowest 8--12 energies and important transition frequencies.
- Convergence with respect to the nonphysical oscillator phase scales.
- Table S1 spectrum and matrix-element comparisons where the paper publishes
  sufficient numerical data; otherwise label results qualitative.
- Same-basis state tracking under small parameter changes, with ambiguities
  surfaced. Cross-cutoff tracking requires explicit tensor embeddings.

### Stage C/D

- All Stage A checks extended to four modes.
- Zero-coupling and high-energy-mode limits.
- \(\epsilon_p\to\infty\) low-energy recovery of S23, including the correct
  constant/gauge treatment.
- \(C_K\to0\) recovery of S20 and \(E_C'\to E_C\).
- Independent convergence of all four mode truncations.
- Comparison with paper spectra, parasitic-mode sidebands, and Table S2
  trends, without claiming numerical reproduction where raw reference values
  are not published.

### Stage E/F

- Exact recovery of the symmetric model at zero asymmetry.
- First-order finite-difference checks of S31 and S32 against a derived full
  asymmetric Lagrangian, if that derivation is adopted.
- State continuity and ambiguity reports through degeneracies/avoided
  crossings.
- Gauge-consistent drive derivatives and explicit conversion from applied
  line amplitude to dimensionless flux.
- Gate metrics evaluated separately from leakage, with fixed logical-state
  conventions.

## 9. Computational scaling and truncation strategy

The primitive dimension is
\(D=N_\Sigma N_\Delta N_S\) for S23 and
\(D=N_\Sigma N_\Delta N_SN_\varphi\) for S20/S30. Dense storage and dense
diagonalization scale as \(D^2\) and \(D^3\) and are not viable as the default.

Initial validation should:

1. build local extended-phase operators,
2. tensor them sparsely in the documented order,
3. request only the low-energy eigenpairs with an iterative Hermitian solver,
4. vary each mode cutoff independently,
5. treat oscillator centers and length scales only as numerical basis
   parameters, without redefining the physical phase coordinates, and
6. record dimension, nonzero count, solve time, residuals, and per-level
   convergence.

For four modes, begin with physically motivated unequal truncations and grow
the mode whose marginal convergence error is largest. A product basis may
eventually need a locally diagonalized or contracted basis, but such a change
must be validated against the direct sparse tensor calculation.

## 10. Flux-drive operator status

S23 unambiguously supplies the two bias derivatives written above. The paper
also discusses phase matrix elements and states that a differential flux port
couples predominantly through particular transformed phase coordinates, but
it does not provide a complete calibrated time-dependent control Hamiltonian
mapping every physical line to \(\phi_{\rm ext}\),
\(\vartheta_{\rm ext}\), or a unique linear combination.

Consequently, the derivatives may be exposed by name, but choosing one as the
physical multimode flux-drive operator requires a documented port/gauge and
mutual-inductance model. That choice remains unresolved.

## 11. Decision lists

### A. Safe to implement directly from source

- Symmetric three-mode Hamiltonian S23, including the printed definition of
  \(E_L'\).
- Canonical S23 coordinates and S24--S25 derived coordinates.
- Table S1 S23 parameter dictionaries, with units recorded as
  \(h\,\mathrm{GHz}\).
- S23 bias derivatives, labeled as derivatives rather than physical drives.
- Symmetric four-mode S20 as its own later model.
- S30's \(C_K\) renormalization of the differential charging coefficient as
  its own later option.
- S31 and S32 as later first-order corrections to S23.
- Sparse tensor construction, sparse eigensolving, independent mode cutoffs,
  cache invalidation, and same-basis state tracking.

### B. Requires a documented modeling choice but can likely be resolved internally

- Numerical basis family and oscillator length scales for the extended S23
  coordinates.
- Convergence thresholds and practical unequal truncation schedules.
- Gauge-consistent comparison between S20/S30 and reduced S23.
- A physical tensor embedding for overlap tracking across different cutoffs.
- Whether to implement S26 after resolving its coefficient discrepancy, and
  which compact/extended mixed basis to use.
- A contracted basis for scaling four-mode calculations, validated against the
  direct sparse tensor basis.

### C. Requires Thomas/author confirmation before claiming physical correctness

- Whether the factor-of-two difference between S23 and S26 is a typo and which
  coefficient is intended.
- The intended corrected form of S21.
- The exact combined four-mode-plus-asymmetry Hamiltonian.
- Whether S31/S32 may be added unchanged to S20/S30, or must be rederived
  before the Born--Oppenheimer reduction.
- The exact physical multimode flux-drive port/operator and its amplitude
  calibration.
- Which external-flux gauge and protected operating point should be canonical
  for later control simulations.
- Quantitative acceptance targets or unpublished numerical references needed
  to declare the multimode model physically validated.

## Source locations

- Gridium paper: Supplementary Note 4, Eqs. S18--S26 and Fig. S4.
- Gridium paper: Supplementary Note 5, Table S1, Figs. S5--S8, and
  Eqs. S29--S32.
- `docs/MODEL_HIERARCHY.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/SIMULATION_CONVENTIONS.md`
- `docs/RESULTS_LEDGER.md`
- `Circuit_Objs/qchard_expgridium.py`
- exploratory three-mode notebook under `Notebooks/`
