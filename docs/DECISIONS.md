# Decisions

## Confirmed decisions

| Date | Decision | Status | Source / provenance |
| --- | --- | --- | --- |
| 2026-09-07 | Develop and simulate a complete single-qubit gate set before transferring control protocols to experiment. | Active | Emiliano/Thomas running notes, p. 1 |
| 2026-09-07 | Begin Gridium control simulations with `IdealGridium` unless explicitly instructed otherwise. | Active | Repository onboarding instructions and Emiliano/Thomas running notes, p. 1 |
| 2026-09-07 | Treat `Simulations/Cphase/` as a structural reference for `Rabi_3Photon`, not as authority for missing single-qubit physics. | Active | Repository onboarding instructions |
| 2026-09-07 | Investigate a three-photon Raman-like process near $\phi_{\mathrm{ext}}=0$ or $\pi$, $\vartheta_{\mathrm{ext}}=\pi$, as a proposed bit-flip direction. Do not treat the protocol as defined until its pathway and control parameters are supplied. | Provisional | Emiliano/Thomas running notes, p. 1; unresolved details listed in `OPEN_QUESTIONS.md` |
| 2026-09-07 | Do not silently change Hamiltonians, units, bases, drives, pulses, simulation behavior, or scientific assumptions. | Active | Repository onboarding instructions |
| 2026-09-07 | Distinguish verified implementation observations from scientific assumptions and record ambiguity rather than guessing. | Active | Repository onboarding instructions |

## Needs verification

- Future scientific decisions require an explicit rationale and an authoritative source or named decision-maker.
- Thomas must define and approve the transition pathway, tone interpretation, logical target, and success criteria before the proposed bit-flip protocol is implemented as settled physics.

## Open questions

- What review process should be used to approve changes that intentionally alter scientific behavior?

## Source / provenance

- Add new entries with a date, decision, status, and traceable source. Do not backfill rationale by inference.
- Scientific project-direction entries above are limited to the supplied Gridium paper and Emiliano/Thomas running notes.
