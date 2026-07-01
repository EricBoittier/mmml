# MMML Documentation

MMML combines molecular mechanics workflows with machine-learned force fields built on JAX.

Use this site for installation, the **CLI reference**, MD workflow guides, and development notes.

## What is here

- [Getting started](getting-started.md) — install with `uv`, serve docs locally
- [CLI overview](cli/index.md) — `mmml commands`, `mmml examples`, tab completion, per-command `--help`
- [md-system YAML configs](md-system-configs.md) — single runs, campaigns, condensed-phase builders
- [MLpot guides](mlpot-settings.md) — settings, medium PBC, spatial MPI, long-range electrostatics
- [Calculator profiling](calculator-profiling.md) — JAX compile vs run, jax-pme primitives, cProfile / TensorBoard
- [PyCHARMM on clusters](pycharmm-mpi.md) — MPI launcher, threading, FFTW
- [PyCHARMM C API: PBC box & pressure](pycharmm-c-api-pbc-box-pressure.md) — KEY_LIBRARY get/set for cell side and CPT pressure tensor
- [Package architecture](package-architecture.md) — module layout and import graph

## External resources

- [Project repository](https://github.com/EricBoittier/mmml)
- [Issue tracker](https://github.com/EricBoittier/mmml/issues)
