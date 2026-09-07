# Repository guidance

This is a scientific-computing repository for superconducting-qubit research. Read `README.md` and the files in `docs/` before changing code.

- Never change scientific behavior silently or invent missing physics.
- Start Gridium control work with `IdealGridium` unless explicitly instructed otherwise.
- Inspect existing implementations before adding new ones. Use `Simulations/Cphase/` as a structural reference where appropriate, not as authority for new physics.
- Preserve units, basis conventions, Hamiltonians, drive functions, pulse logic, scientific assumptions, and existing behavior unless the task explicitly requires a change.
- Distinguish verified code observations from physics assumptions. Record unresolved scientific ambiguity instead of guessing.
- Keep changes within the requested scope and run relevant tests afterward. Report pre-existing failures separately.
