# Gridium simulation research

This repository supports superconducting-qubit research on Gridium. The current objective is to develop complete single-qubit control in simulation before transferring that control to experiment, beginning with the `IdealGridium` model.

Repository map:

- [`Circuit_Objs/`](Circuit_Objs/) contains circuit models, coupled-system utilities, pulse shapes, and evolution helpers.
- [`Simulations/`](Simulations/) contains simulation workflows. [`Simulations/Cphase/`](Simulations/Cphase/) is a structural reference for the developing [`Simulations/Rabi_3Photon/`](Simulations/Rabi_3Photon/) workflow.
- [`Tests/`](Tests/) contains model and simulation tests.
- [`docs/`](docs/) contains project context, conventions, decisions, open questions, and research-result provenance.
- [`environment.yml`](environment.yml) records the Conda environment and Python dependencies.

Read [`AGENTS.md`](AGENTS.md) before making changes and consult the documents in [`docs/`](docs/) for deeper context.
