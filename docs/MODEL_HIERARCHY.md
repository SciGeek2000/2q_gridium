# Model hierarchy

## Confirmed information

- `IdealGridium` in `Circuit_Objs/qchard_idealgridium.py` is described as a one-mode toy model and is the required starting point for Gridium control work.
- `ExpGridium` in `Circuit_Objs/qchard_expgridium.py` is described as a detailed experimental multi-mode model. Its implementation is currently incomplete and its docstring says it should be three-mode.
- Additional circuit objects include Fluxonium, Transmon variants, Squid, cavity/resonator, and two-level-system models.
- `CoupledObjects` composes individual circuit objects and coupling terms into a joint system.
- `AbstractQubit` records a shared interface, although the inspected model classes are not declared as subclasses of it.

## Needs verification

- Which class or future implementation represents the planned four-mode asymmetric Gridium model.
- Whether `ExpGridium` is the intended foundation for that model.
- Which model interfaces are stable requirements versus provisional compatibility conventions.

## Open questions

- What validated milestone should trigger moving control studies beyond `IdealGridium`?
- How should model assumptions and provenance be versioned as the hierarchy evolves?

## Source / provenance

- `Circuit_Objs/qchard_abstractobj.py`
- `Circuit_Objs/qchard_idealgridium.py`
- `Circuit_Objs/qchard_expgridium.py`
- `Circuit_Objs/qchard_coupobj.py`
- Remaining model modules under `Circuit_Objs/`.
