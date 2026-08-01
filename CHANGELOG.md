# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [PEP 440](https://peps.python.org/pep-0440/) pre-release
identifiers rather than strict [SemVer](https://semver.org/) while the project
is in alpha (`0.x`). See [`docs/releasing.md`](docs/releasing.md) for the tag
and versioning process.

## [Unreleased]

### Added

- Docs: student walkthrough for trainable hybrid MM LJ scales
  ([`docs/hybrid-mm-lj-scales.md`](docs/hybrid-mm-lj-scales.md)).
- Hybrid MM: learnable per-CGenFF-type LJ σ and ε scales (`--learn-mm-lj-scales`)
  for MIC hybrid training; scales persist in `hybrid_mm.json` and load into MD
  `ep_scale` / `sig_scale` (`--mm-lj-scales-file` or auto next to checkpoint).
  See `docs/hybrid-mm-charges.md` and `examples/hybrid_mm_charges/train_fixed_lj_scales.yaml`.
- `md-system` `interaction_policy: ./policy.yaml`: config-relative path
  resolution, load/validate on all runners, fail-closed for multi-provider /
  near–far policies, manifest provenance. See `docs/md-interaction-policies.md`.
- Batched pure-ML distance umbrella sampling (`mmml umbrella-sample`) and
  CV MBAR post-processing (`mmml umbrella-mbar`): pack K restrained copies into
  one PhysNet/SpookyNet batch, NVT via JAX-MD Langevin by default (Nose-Hoover
  optional), optional Hamiltonian replica exchange (`--replica-exchange`), then
  pymbar. Seeding fixes `atom_i` and can rigidly translate `--move-with` groups
  (default `dt=0.1` fs). Exports CoM-centered window XYZs (optional) and an ASE
  `umbrella_bin_minima.traj` of the lowest `E_ML+W` frame per window. See
  `docs/umbrella.md`.
- First-class CLI: `mmml compare-charmm-ml` (CHARMM PSF charges vs joint
  PhysNet/DCMNet dipoles and ESP on a validation split).

- CI test-shape gates: `scripts/ci/check_test_report.py` reads the JUnit XML each
  pytest step now emits and fails when too few tests passed, too many skipped, a
  failure was recorded, or the report is missing entirely. `pytest`'s exit code
  alone could not distinguish a passing suite from one that never ran. Also
  `scripts/ci/assert_pycharmm_live.py` (hard PyCHARMM import before the live
  job), `make test-shape`, and a 60-minute timeout on the `build` job.
- Repo-wide guard against the unit-constant bug class
  (`tests/unit/test_conversion_constant_drift.py`). Every bug in the units audit
  was a module-local literal used on both the write and the read side, so it
  round-tripped perfectly and disagreed only with physics — invisible to any
  test of the module holding it. The guard parses the package with `ast` (no
  imports, so it also covers modules needing JAX/PySCF/CHARMM) and asserts that
  every module-level constant reusing a `mmml.data.units` name agrees with the
  canonical value, and that ~25 conversions with no canonical twin match an
  SI/CODATA derivation spelled out in the test. Tolerance 1e-4: rounded literals
  in the tree deviate by at most 1.5e-5, the historical `1.88873` transposition
  by 5.3e-4.
- CI coverage floor: `scripts/ci/check_coverage_floor.py` plus a `Coverage
  floor` step and `make coverage-gate`. A floor, not a target — ~18.7k
  statements need live CHARMM, ~10.9k are plotting and ~5.3k need
  PySCF/torch/GPU, so the CI-reachable ceiling is near 70%. It also pins the
  absolute covered-line count, since deleting untested code raises the
  percentage. Complements the Codecov status, which is advisory and needs a
  token.
- Oversized-function ratchet (`tests/unit/test_oversized_function_ratchet.py`),
  two tiers: the 11 functions over 1,000 lines may not grow and none may be
  added; the 26 over 500 lines are capped by count, so ordinary edits to an
  already-large function do not turn the suite red but a 27th does.
  `run_staged_workflow` alone holds 699 of the 902 uncovered statements in its
  module and caps that file near 35% coverage; the ratchet keeps the pattern
  from spreading while decomposition waits on the golden-record harness.
- `MMML_DISABLE_CHARMM=1` makes CHARMM discovery report nothing and blocks
  `import pycharmm` outright, so `make test-ci` genuinely reproduces the
  libcharmm-free CI environment. Setting `CHARMM_LIB_DIR` to a nonexistent path
  does not work: a lib-less explicit override is treated as stale and replaced
  by the discovered `setup/charmm` tree.

### Fixed

- **`md-system` died in the child process on both PBC backends.** `run_sim`
  grew `--hybrid-hamiltonian` / `--shared-cutoff` and `md_system.build_command`
  forwards them to every backend unconditionally, but neither
  `md_pbc_suite.ase` nor `md_pbc_suite.jaxmd` declared them — so the forwarded
  argv hit argparse exit 2 *inside the subprocess*, after the run had started.
  Both parsers now accept them and thread them into `setup_calculator` and
  `CutoffParameters`, matching `run_sim`. `md_pbc_suite/jaxmd.py::main` grew a
  `build_parser` (its 743-line argparse block, extracted) so the argv a backend
  receives can be parsed in a test rather than only in a live run;
  `tests/unit/test_md_system_ase_cmd.py` now parses the *whole* forwarded argv
  against both backends instead of one flag at a time.
- `tests/unit/test_lambda_jaxmd_neighbors.py` aborted collection wherever
  libcharmm is absent — `lambda_jaxmd` imports `lambda_dynamics`, which does a
  module-level `import pycharmm.param`. pytest reports that as "Interrupted: 1
  error during collection" and runs **zero** tests, so it failed the whole
  build rather than skipping one file. Guarded with a module-level skip;
  deferring that import in `lambda_dynamics` would let the tests run in CI.
- `mmml pes-design` was registered without a `CLI_NAV_GROUPS` entry, which made
  `scripts/generate_cli_docs.py` refuse to run at all and left the generated
  CLI reference and package-architecture docs stale in CI.
- **DCMNet dipole units.** `dcmnet/loss.py:pred_dipole` multiplied by `1.88873`
  and documented its result as Debye. The value is a transposed-digit typo for
  the Angstrom -> bohr factor `1.8897261` (5.3e-4 relative), and the unit was
  never Debye — both callers in `dcmnet/analysis.py` convert the residual with
  `au_to_debye` afterwards. The docstring now states atomic units (e*bohr) and
  the factor comes from `mmml.data.units.ANGSTROM_TO_BOHR`.

  `dcmnet_ase.DCMNetCalculator._compute_molecular_dipole` carried the same
  literal under an "atomic units to Debye" comment while its input was
  e*Angstrom, so **every dipole that calculator reported was ~2.54x too small**
  despite being labelled Debye in the method docstring, in `get_dcm_data`, and
  in the example script's printout. It now applies `EANGSTROM_TO_DEBYE`.

  `dcmnet/analysis.py` held a third independent literal for e*bohr -> Debye;
  it now uses the shared `EBOHR_TO_DEBYE`, which `mmml.data.units` derives from
  the other two so the chain cannot drift apart again. `au_to_kcal` likewise
  moved to `HARTREE_TO_KCAL_MOL` (`627.509` -> `627.509474`).

  Impact: recorded DCMNet dipole MAEs shift by 5.3e-4 relative; checkpoints are
  unaffected (the change is to a reported/loss scale, not to parameters). ASE
  calculator dipoles change by a factor of 2.5417. Covered by
  `tests/unit/test_dcmnet_dipole_units.py`, which anchors on CODATA values
  computed in the test and on the identity
  `EANGSTROM_TO_DEBYE == ANGSTROM_TO_BOHR * EBOHR_TO_DEBYE`.
- **PhysNetJAX `cut_vdw`.** The DCMNet copy of `cut_vdw` was fixed during the
  units audit; the PhysNetJAX copy in `physnetjax/data/cut_grid.py` kept the
  same defect — `elements` stayed a plain list on the element-symbol path, so
  `elements[closest_atom]` raised "only integer scalar arrays can be converted
  to a scalar index" for exactly the input the docstring advertises.
  `physnetjax/data/data.py` also called `cut_vdw` without importing it, so
  `prepare_multiple_datasets(..., esp_mask=True)` raised `NameError` for every
  caller. Both fixed; `tests/unit/test_physnetjax_cut_grid.py` pins the two
  implementations to identical output so a fix to one cannot skip the other.
- Docs: `docs/UNITS_SUMMARY.md` listed the E-field PhysNet Coulomb prefactor
  `7.199822675975274` as an unresolved question. It is `(e²/4πε₀)/2` — halved
  because the pair sum runs over ordered pairs — now named
  `COULOMB_PAIR_FACTOR_EV_A` and anchored against `1/(4πε₀)` in the drift test.
- Test isolation: `test_mpi_openmpi_static_shmem_fallback` leaked
  `LD_PRELOAD` / `DYLD_INSERT_LIBRARIES` into the real environment, pointing at a
  library under a `tmp_path` pytest later deleted. Every subsequent test that
  spawned a subprocess then died in the dynamic loader (exit -6) with a message
  naming neither the cause nor the culprit; six unrelated tests failed that way
  in a full-suite run while passing in isolation.
- `python -m mmml.data.npz_schema` raised `NameError` instead of printing usage:
  `sys` was imported inside `main()` only, but the module-level guard calls
  `sys.exit(main())`.
- Codecov could not report a regression: `patch: false` waived coverage on new
  code entirely and a 50-percentage-point project threshold let total coverage
  halve while the status stayed green.
- Missing comma in `mmml/data/qcml/atomic_reference_energies.json` that broke
  `json.load` (and any import of `mmml.data`) after the QCML reference table
  update.

## [0.1.0a2] - 2026-07-27

Second tagged alpha. Focuses on enhanced-sampling / reaction-coordinate
workflows, PyCHARMM-interface robustness, and CLI ergonomics on top of the
`0.1.0a1` baseline.

### Added

- Nudged Elastic Band (NEB) support with accompanying documentation.
- Diffusion Monte Carlo (DMC) CLI commands and documentation.
- ADUMB / umbrella-sampling reaction-coordinate tooling: RC distance walls
  with preflight checks and automatic reinstallation during overlap-chunk
  recovery, `RESDistance` / single-line NOE distance walls, MMFP wall setup
  under MPI, and bond-difference (ξ) reaction-coordinate handling.
- New CLI commands and flags: `npz2traj` (convert NPZ datasets to ASE
  trajectories), `--from-pdb` full-system cold start, `--mm-pair-source` for
  selecting the MM pair provider on `jax_mic` hybrids, and `make-box` solvent
  density handling.
- PyCHARMM interface: pre-dynamics CHARMM lingo scripting, execution/splitting
  of lingo scripts, extra RTF/PRM path support in CGenFF residue handling, and
  PDB parsing helpers for residue handling.
- Geometry / frame-selection helpers for reaction coordinates (selection by
  `r_ClC` / `r_CN` / ξ, mass-weighted centering for PDB output).

### Changed

- JAX backend/device management in the PyCHARMM interface is more robust:
  CPU/GPU backend availability is detected explicitly, and default GPU runs no
  longer defer XLA GPU warmup (avoiding spurious "CPU backend is not
  registered" warnings). Deferral is now gated on the CPU-load path.
- `charmm_io_staging_root` now creates per-user subdirectories.
- Documentation: Read the Docs / MkDocs configuration, NEB and DMC guides, and
  refreshed ADUMB (NH3–CH3Cl) examples.

### Fixed

- Numerous PyCHARMM dynamics, restraint, and overlap-guard fixes; regenerated
  CLI reference and package-architecture docs to match the current tree.

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

[Unreleased]: https://github.com/EricBoittier/mmml/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/EricBoittier/mmml/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/EricBoittier/mmml/releases/tag/v0.1.0a1
