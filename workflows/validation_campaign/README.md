# MMML validation campaign

This workflow is the campaign index for proving that supported MM, ML, and
hybrid MM/ML simulations work across liquids, gas-phase peptides, and solvated
peptides. It does not treat a submitted job as success: every task has explicit
acceptance checks and a proof directory.

## Scientific goals

| Goal | Systems | Required proof |
|---|---|---|
| `platform` | every compute environment | environment report, JAX x64/CUDA status, import and tiny-energy checks |
| `force_energy` | representative monomers/dimers | component energies, LJ/electrostatic audit, finite differences, cutoff scans |
| `backend_parity` | DCM:10 and TIP3:small | identical-state NVE/NVT traces, drift metrics, round-trip state audit |
| `pure_liquids` | BENZ, TIP3, DCM, ACO | preparation audit, stable NVT/NPT, density/RDF/temperature/energy plots |
| `peptide_gas` | alanine, trialanine | minimized structure, finite-difference forces, NVE/NVT stability and conformational plots |
| `peptide_solution` | alanine+TIP3, trialanine+TIP3 | solvation audit, stable NVT/NPT, peptide-water RDF/contact and conformation plots |
| `coverage` | all supported combinations | machine-readable method/backend support matrix with pass, fail, blocked, unsupported |

Each task writes under
`artifacts/validation_campaign/<run_id>/<environment>/<task/path>/`:

- `request.json`: immutable task/environment/config/checkpoint request;
- `stdout.log`, `stderr.log`, and scheduler metadata;
- `status.json`: terminal state and exact failure, if any;
- `metrics.json`: quantitative acceptance metrics;
- `proof.json`: acceptance checks and source artifact hashes;
- `plots/`: human-readable figures using the repository plot style;
- `provenance.json`: git revision, dirty diff hash, Python/JAX/CUDA/CHARMM details.

Run directories accumulate; a new run never overwrites an older receipt.
`status` reads the newest receipt per (task, environment).

## How proof works

A task is `PASS` **only** when every acceptance check declared in
`campaign.yaml` is present in its `proof.json` and true. This is the one
invariant of the campaign:

- Only `exit_zero` may be certified by the harness, which observes the process
  exit status directly (`finalize_task.py`).
- **Every other check must be asserted by a scientific driver.** No component of
  this harness may write a metric it did not compute.
- A job that was submitted, or that exited zero without writing the checks it
  promised, is `INCOMPLETE` — never `PASS`.

| State | Meaning |
|---|---|
| `PASS` | every declared acceptance check is present and true |
| `FAIL` | a declared check ran and was false |
| `BLOCKED` | a known defect prevents the task from running |
| `GATED` | a prerequisite task has not passed |
| `NEEDS_DRIVER` | catalogued, but its scientific driver is not built yet |
| `INCOMPLETE` | no receipt, or proof missing for a declared check |

`prepare`, `submit`, and `run-local` refuse to dispatch `blocked`, `gated`, or
`needs_driver` tasks unless `--include-not-ready` is passed. This is deliberate:
dispatching a task with no driver would produce a receipt that proves nothing.

## Environments

- `pcbach`: CPU-heavy and memory-heavy reference generation, cache building,
  classical MM, analysis, and large scans.
- `scicore`: primary GPU production and broad checkpoint/method matrices.
- `pcstudix`: fast smoke tests and focused GPU diagnostics; do not launch the
  full matrix until the handoff gates pass.
- `local_laptop`: CPU debugging, dry-runs, unit tests, plots, and collection.
- `local_computer`: CUDA debugging and single-GPU integration tests.

Environment definitions live in `environments/*.yaml`. Cluster launch uses
`scripts/submit_slurm.py`; local launch uses `scripts/run_local.py`. Both write
the same request/provenance/status layout, so a laptop receipt and a cluster
receipt are indistinguishable to `status` — only proven and unproven differ.

Generated Slurm scripts are **repo-relative** and `cd` to the environment's own
`repo_root`, so a script rendered on the laptop runs correctly on the cluster.
`scripts/pc_bach_env.sh` is sourced automatically for `pcbach`.

Thin per-environment entry points live in `launch/`:

```bash
workflows/validation_campaign/launch/local_laptop.sh --tier static
workflows/validation_campaign/launch/pcbach.slurm.sh prepare --tier static
```

## What is proven today

Run `campaign.py status` for the live answer. As of the last run, only
`platform.static` and `force_energy.unit` are `PASS`, and only on
`local_laptop`. Everything under `pure_liquids`, `peptide_gas`, and
`peptide_solution` is `NEEDS_DRIVER`: the systems, methods, and acceptance
checks are catalogued, but the scientific drivers that would compute them are
not written yet. They currently dispatch to `scripts/task_placeholder.py`, which
writes `NEEDS_DRIVER` and exits 2 rather than inventing a result.

Building those drivers is the next piece of work. Each one must write a
`proof.json` whose checks it actually computed.

## Commands

For the maintained pc-studix calculator/backend smoke matrix, including
PhysNet, SpookyNet, learned MBD/multipoles, EField, xTB, PySCF, DFTB3-D4,
PyCHARMM, JAX-MD, rigid MC, charge modes, and long-range solvers, see
[`PCSTUDIX_SMOKE_MATRIX.md`](PCSTUDIX_SMOKE_MATRIX.md).

List the campaign and current proof state:

```bash
python workflows/validation_campaign/scripts/campaign.py list
python workflows/validation_campaign/scripts/campaign.py status
```

Generate (but do not submit) Slurm scripts:

```bash
python workflows/validation_campaign/scripts/campaign.py prepare \
  --environment pcstudix --tier smoke
```

Submit ready tasks after inspecting the generated scripts:

```bash
python workflows/validation_campaign/scripts/campaign.py submit \
  --environment pcstudix --tier smoke
```

Run a local tier:

```bash
python workflows/validation_campaign/scripts/campaign.py run-local \
  --environment local_laptop --tier static
```

Campaign state is summarized to `artifacts/validation_campaign/summary.{json,md}`:

```bash
python workflows/validation_campaign/scripts/campaign.py status --write
```

## Current hard blocker

`backend_parity.dcm10_pycharmm_jaxmd` is blocked at the PyCHARMM velocity
handoff boundary. Text/native restart attempts start at 0 K, while direct
`dynamics_run_kw(init_velocities=...)` currently SIGSEGVs in `dynopt` due to
the gfortran `bind(c)` assumed-shape-array ABI. This blocker must remain visible
in the summary and must gate solvent-burst production that alternates backends.
