# Theory

## Source boundary

Scientific statements in this document are limited to:

- [The superconducting grid-states qubit](<references/2509.14656v1 (3).pdf>) (the Gridium paper).
- [Emiliano and Thomas Gridium Running Notes](<references/Emiliano and Thomas Gridium Running Notes.pdf>).

Repository implementation observations are labeled separately and are not treated as scientific authority.

## Confirmed information from the Gridium paper

### Ideal and extended one-mode Hamiltonians

The paper defines the ideal doubly periodic Gridium/GKP Hamiltonian as

$$
\hat H_{\mathrm{GKP}}=-E_S\cos(2\pi\hat n)+E_{2J}\cos(2\hat\phi),
$$

where $\hat n=\hat q/(2e)$, $[\hat\phi,\hat n]=i$, $E_S$ is the coherent quantum-phase-slip amplitude, and $E_{2J}$ is the effective Cooper-quartet-tunneling energy. The practical single-mode model adds unavoidable quadratic confinement:

$$
\hat H=E_C\hat n^2+\frac{1}{2}E_L(\hat\phi+\phi_{\mathrm{ext}})^2
-E_S\cos(2\pi\hat n)+E_{2J}\cos(2\hat\phi).
$$

For charge-dispersion calculations the paper makes the substitution $\hat n\rightarrow\hat n+n_g$. It identifies the Gridium regime as

$$
E_{2J},E_S\gg E_C,E_L,
$$

so that the quadratic corrections are small. The quadratic terms make the states normalizable and place a finite Gaussian envelope around the phase-space grid. (Gridium paper, main text Eq. 1-2 and PDF pp. 1-3; Supplementary Note 1, PDF pp. 11-12.)

The paper studies two representative one-mode parameter sets, in $h\cdot\mathrm{GHz}$ units ordered as $[E_{2J},E_S,E_C,E_L]$: $[12,4,0.5,0.5]$ and the more protected $[12,4,0.1,0.1]$. These are examples, not a universal definition of "soft" and "hard."

### Symmetric and protected operating points

- The one-mode spectrum is $\pi$-periodic in $\phi_{\mathrm{ext}}$ and its eigenstates become pairwise degenerate at $\phi_{\mathrm{ext}}=0$ and $\pi$.
- In the physical circuit, the Gridium ($d=2$) regime is obtained at KITE flux $\vartheta_{\mathrm{ext}}=\pi$.
- The running notes call $(\phi_{\mathrm{ext}},\vartheta_{\mathrm{ext}})=(0,\pi)$ and $(\pi,\pi)$ protected points. The paper also refers to these as doubly symmetric flux biases and experimentally characterizes the $[\pi,\pi]$ point for an inductively shunted CQT prototype.
- Figure 1c-d illustrates the two lowest one-mode wavefunctions at $\phi_{\mathrm{ext}}=\pi/2$. That illustration is not itself the paper's statement of a degeneracy point.

(Gridium paper, main text pp. 2, 5-6; Supplementary Notes 1, 3, 5, and 10, PDF pp. 11, 16, 18-22, and 28; running notes, p. 1.)

### Computational states

The paper identifies the two lowest states, written as $|\psi_0\rangle$ and $|\psi_1\rangle$ in the main text and as $|0\rangle$ and $|1\rangle$ in the supplement, as the computational pair. They are coherent superpositions of alternating peaks in both phase and charge, with the parity structure responsible for cancellation or suppression of local linear couplings. At the symmetric biases they belong to a pairwise-degenerate low-energy subspace. The sources do not specify a unique logical phase convention, logical Pauli operators, or how the basis should be fixed numerically at exact degeneracy.

### Charge and phase selection rules

At $\phi_{\mathrm{ext}}=0$, Supplementary Note 1 reports:

- Phase matrix elements have a checkerboard pattern: `|0>` phase-couples to `|1>`, `|3>`, `|5>`, and so on; `|1>` phase-couples to `|0>`, `|2>`, `|4>`, and so on.
- Charge coupling has a similar pattern, except that charge coupling vanishes between degenerate partners: `|0>` to `|1>`, `|2>` to `|3>`, `|4>` to `|5>`, and so on.
- In the $[12,4,0.5,0.5]$ example, the phase matrix element between `|0>` and `|1>` remains finite at degeneracy. In the more protected $[12,4,0.1,0.1]$ example, phase matrix elements between degenerate partners become exponentially small.
- In that more protected example, only the `0-to-5` transition retains appreciable charge coupling throughout the full flux interval examined.

This supplies an important qualification to the paper's broad symmetry-protection description: finite quadratic corrections can leave a finite `0-to-1` phase matrix element, while reducing those corrections suppresses it.

### Protection and addressability

Increasing $E_{2J}/E_C$ expands charge-basis grid support and suppresses charge dispersion. Lowering the relevant inductive energies expands phase-basis grid support and suppresses flux dispersion and the `0-to-1` phase matrix element. As the circuit approaches the ideal Hamiltonian, the computational pair becomes less sensitive to local charge and flux perturbations.

The same trend reduces low-order transition matrix elements. The paper reports that more-protected devices have fewer observable transitions, nearly uniform dispersive shifts, and a computational subspace that cannot be probed directly at the protected point. Thus stronger protection makes ordinary dipole control and readout progressively harder. This is a protection-addressability tradeoff, not evidence that every possible control protocol is forbidden. (Gridium paper, main text pp. 5-6; Supplementary Notes 1 and 5, PDF pp. 11-12 and 18-22.)

## Repository observations

- `Circuit_Objs/qchard_idealgridium.py` implements a one-mode `IdealGridium` object. Its scientific interpretation must remain tied to the equations and conventions above unless explicitly reviewed.
- The implementation and the paper should not be assumed identical term-by-term beyond relationships that have been checked explicitly.

## Needs verification

- The precise logical-basis convention at exact degeneracy.
- Which drive operator is intended for each proposed control tone.
- Whether the paper's selection rules remain exact or only approximate in the intended simulation after finite truncation, disorder, or asymmetry.
- Whether the `0-to-5` transition participates in the proposed three-photon bit flip; the paper establishes its charge coupling and readout relevance, not that control pathway.

## Source / provenance

- Gridium paper: main text Eq. 1-2 and Fig. 1; Supplementary Notes 1, 4, and 5.
- Running notes: one-page project and control summary.
