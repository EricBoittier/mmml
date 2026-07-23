# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [PEP 440](https://peps.python.org/pep-0440/) pre-release
identifiers rather than strict [SemVer](https://semver.org/) while the project
is in alpha (`0.x`). See [`docs/releasing.md`](docs/releasing.md) for the tag
and versioning process.

## [Unreleased]

## [0.1.0a1] - 2026-07-22

First tagged alpha. Pre-alpha development happened directly on `main` without
tags, so this entry summarizes the state of the project at the tag rather
than listing every prior commit.

### Highlights

- Hybrid ML/MM molecular dynamics via `mmml md-system`, with ASE, JAX-MD, and
  PyCHARMM backends, and species-aware monomer/pair interaction ownership
  (see [`docs/md-interaction-policies.md`](docs/md-interaction-policies.md)).
- Structure/box building (`make-res`, `make-box`, `build-crystal`,
  `liquid-box`), quantum-chemistry data generation (`pyscf-*`), and PhysNet /
  DCMNet / SpookyNet / external-electric-field model training and evaluation
  (`physnet-train`, `physnet-evaluate`, `train-joint`, `efield-train`,
  `efield-evaluate`).
- MD analysis tooling: IR/VCD/Raman spectra from trajectories
  (`mmml.spectra`), vibrational-mode / finite-difference validation
  (`mode-check`), and a capability-aware smoke-test matrix for scientific
  validation campaigns (`mmml.validation.smoke_matrix`).
- A FastAPI + web molecular viewer (`mmml gui`).
- `mmml doctor` / `mmml health-check` / `mmml env` for verifying JAX, CHARMM,
  and Packmol readiness before running a simulation.

### Changed

- CLI reference docs (`docs/cli/commands/`) are now generated and CI-checked
  from `mmml/cli/registry.py`; run `scripts/generate_cli_docs.py` after any
  CLI flag/command change (see `CLAUDE.md`).

### Removed

- Removed the long-deprecated `train`, `evaluate`, `ef-train`, `ef-evaluate`,
  and `ef-md` CLI commands. `train` and `evaluate` never did real model
  training/evaluation (they prepared batches or fabricated metrics from
  noised targets and reported success regardless); use `physnet-train` /
  `train-joint` and `physnet-evaluate` / `efield-evaluate` instead. `ef-train`
  / `ef-evaluate` / `ef-md` are superseded by `efield-train` / `efield-evaluate`
  / `efield-md`.

### Known limitations

- Interfaces are still settling and may change without notice ahead of a
  first stable (non-alpha) release.
- Dependency versions are loosely pinned outside of a few git-pinned extras
  (`jax-md`, `jax-pme`); expect to need `uv lock --upgrade` occasionally.
- Test coverage is uneven across subpackages; `mmml/gui`'s FastAPI routes and
  the OpenGL/OpenXR viewer, and parts of `mmml/interfaces/pycharmmInterface`
  that require a live PyCHARMM/MPI runtime, are exercised primarily by manual
  and CI-only (`tests/charmm_mpi/`) testing rather than by the default unit
  suite.

[Unreleased]: https://github.com/EricBoittier/mmml/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/EricBoittier/mmml/releases/tag/v0.1.0a1
