# Simulation conventions

## Confirmed information

- Circuit classes generally expose energy/unit labels as GHz by default.
- Current workflow plots and status messages label time in ns.
- The CPhase and Rabi workflow helpers construct time grids with `2 * int(T_gate) + 1` points.
- `qchard_evolgates.py` multiplies the no-drive Hamiltonian by `2 * pi` in the inspected propagator helpers. This is an implementation observation and must not be changed without scientific review.
- `CoupledObjects` documents its default basis as the tensor product of the individual objects' eigenbases.
- CPhase configuration uses string state labels such as `"00"`; the single-qubit Rabi prototype uses integer level indices.
- Pulse and system settings are stored under each workflow's `yamls/` directory.

## Needs verification

- The canonical relationship between stored energy values, angular frequencies, ordinary frequencies, and time units across every workflow.
- Required basis ordering and state-label conventions for new single-qubit-control results.
- Which numerical cutoffs and solver tolerances are validated for each model and task.

## Open questions

- Which conventions are scientific requirements versus historical implementation choices?
- What metadata must accompany figures and saved simulation results?

## Source / provenance

- `Circuit_Objs/qchard_coupobj.py`
- `Circuit_Objs/qchard_evolgates.py`
- `Circuit_Objs/qchard_idealgridium.py`
- `Simulations/Cphase/workflow_funcs.py`
- `Simulations/Rabi_3Photon/workflow_funcs.py`
