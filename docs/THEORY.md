# Theory

## Confirmed information

- `Circuit_Objs/qchard_idealgridium.py` describes `IdealGridium` as a one-mode toy model.
- `Circuit_Objs/qchard_expgridium.py` describes `ExpGridium` as an experimental multi-mode model and states that it should be a three-mode model.
- Hamiltonians and operators are implemented in the circuit-object modules; this document does not reinterpret them.

## Needs verification

- The derivation, approximation regime, and primary source for every Gridium Hamiltonian term.
- The exact symmetry and protection assumptions associated with each model.
- Which statements in code comments remain current scientific assumptions.

## Open questions

- What is the canonical theoretical definition of the protected operating point used by this project?
- How should the intended four-mode asymmetric model relate to the current `ExpGridium` implementation?

## Source / provenance

- Code observations: `Circuit_Objs/qchard_idealgridium.py` and `Circuit_Objs/qchard_expgridium.py`.
- The model files contain inline literature references; those sources have not been independently summarized or validated here.
