# Rabi_3Photon validation artifacts

This directory contains durable evidence from frozen, serious
`IdealGridium` X90 and X180 validation runs. The control model couples all
tones through the abstract global phase coordinate `IdealGridium.phi()`.
The historical YAML value `drive_type: flux` does not identify a calibrated
experimental flux line.

For each target gate:

- `*_state_populations.png` shows energy-eigenstate populations for logical
  inputs `|0>` and `|1>`, including the `0-5-4-1` pathway levels.
- `*_intermediate_populations.png` expands the vertical scale for `|4>`,
  `|5>`, and the other nonlogical retained levels with the largest transient
  populations in the simulated trajectories.
- `*_leakage.png` shows state-resolved and logical-input-average leakage.
- `*_drive_traces.png` shows the three actual scalar coefficients passed to
  the time-dependent solver, including amplitude scaling.
- `*_convergence.png` summarizes solver/sampling, retained-level, and LC-cutoff
  validation using absolute gate infidelity, `1 - F_gate`, alongside leakage.
- `*_summary.csv` is a one-row reproducibility and Pareto-analysis record.
- `*_timeseries.csv` contains populations, leakage, and drive coefficients at
  every saved time.
- `*_convergence.csv` contains the numerical convergence table.

`validated_gate_runs.csv` combines the per-gate summary rows so fidelity,
duration, and leakage tradeoffs can be plotted without rerunning simulations.

Regenerate the frozen artifacts from the repository root with:

```shell
python -m Simulations.Rabi_3Photon.generate_validation_artifacts
```

The generator reruns validation from the frozen experiment YAMLs; it does not
re-optimize either gate.
