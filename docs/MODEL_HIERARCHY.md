# Model hierarchy

## Source boundary

Scientific model relationships below come only from the [Gridium paper](<references/2509.14656v1 (3).pdf>) and the [Emiliano/Thomas running notes](<references/Emiliano and Thomas Gridium Running Notes.pdf>). Class and directory names are repository observations.

## Confirmed progression

### 1. IdealGridium / one-mode extended GKP model

- The paper uses the one-mode Hamiltonian with small quadratic charge and phase confinement to isolate the essential Gridium behavior.
- It is a representative low-energy model rather than a complete physical-circuit description.
- The running notes explicitly require control simulations to start with `IdealGridium`.
- The paper reports that agreement between the one-mode and physical-circuit spectra extends to higher frequencies as the device approaches the ideal energy-ratio regime.

### 2. Low-energy three-mode circuit model

- Starting from the full circuit, the paper eliminates a high-energy mode with a Born-Oppenheimer approximation when the cross-circuit stray capacitance $C_p$ is vanishingly small, yielding the three-mode Hamiltonian in Eq. S23.
- After a coordinate change, Eq. S26 exposes the compact variable and offset charge. The three-mode wavefunction forms a two-dimensional grid for suitable parameters; a slice along one coordinate resembles the one-dimensional grid of the extended one-mode model.
- For the devices analyzed in the paper, the three-mode model agrees well with measured spectra below about 5 GHz. That frequency is an observation for those devices, not a universal cutoff.

### 3. Four-mode circuit model

- The paper's Eq. S20 retains the additional mode associated with finite $C_p$. Eq. S30 further includes cross-KITE capacitance $C_K$ through a renormalized differential-mode charging energy.
- The paper says a four-mode model is needed for the studied device when accurately modeling transitions above roughly 5 GHz or when higher excited states and sideband transitions involving array modes matter.
- Reported fitting cost grows from minutes for the one-mode model, to days for the three-mode model, to as much as two weeks for the four-mode model.

### 4. Asymmetry

- The paper treats Josephson-energy and inductive asymmetry as explicit perturbations to the symmetric three-mode Hamiltonian (Eqs. S31-S32). Simulated asymmetry changes some frequencies and amplitudes and can relax suppression of single-Cooper-pair tunneling, while the overall spectral structure and selection rules remain largely consistent for the cases studied up to 10% asymmetry.
- The running notes identify a "four mode asym gridium" as the later simulation target.
- The supplied sources do not explicitly define one combined four-mode asymmetric Hamiltonian or map it to a repository class. That relationship must not be inferred.

## Repository observations

- `Circuit_Objs/qchard_idealgridium.py` contains the current one-mode `IdealGridium` class.
- `Circuit_Objs/qchard_expgridium.py` is described in the repository as an incomplete experimental multimode model, but the supplied scientific sources do not establish whether this class is the canonical three-mode or future four-mode asymmetric implementation.
- `CoupledObjects` and other circuit classes are implementation infrastructure, not stages in the paper's Gridium reduction by themselves.

## Transition criteria that remain open

- What validation milestone permits moving control work from `IdealGridium` to the three-mode model?
- Is the three-mode model required before the four-mode asymmetric model, or is the intended project workflow one-mode directly to four-mode?
- Which parasitic modes, capacitances, and asymmetries must be present in the eventual four-mode simulation?
- What frequency range and excited-state content must each model reproduce for control work?

## Source / provenance

- Gridium paper: Supplementary Notes 4, 5, and 11 (PDF pp. 17-23 and 29-30).
- Running notes: instruction to start with `IdealGridium` and later move to a four-mode asymmetric Gridium model.
