# Environment & clusters

Getting MMML to run *where* you need it: resolving checkpoints and CHARMM paths,
diagnosing a broken environment, and the MPI / threading / launcher details that
decide whether a cluster job is fast or merely running.

## Happy path

```bash
mmml doctor                    # is this machine ready? (JAX, CHARMM, Packmol)
mmml env                       # resolved + bundled checkpoints, CHARMM paths
mmml env --json                # same, parseable
mmml health-check --require-gpu
```

If `doctor` is clean but MLpot still misbehaves under MPI:

```bash
mmml mpi-check                 # validate OpenMPI / CHARMM / mpi4py
mmml mpi-launch ...            # launch with an explicit JAX execution policy
```

## Preflight runbook

Use `doctor` for install readiness and `health-check` for run-specific interface
readiness. They share the same probe implementation, but their intent is
different:

| Command | Use when | What it proves |
|---|---|---|
| `mmml doctor` | setting up a clone or checking a CI/cluster image | core Python imports, JAX, GPU-quantum dependency warnings, CHARMM paths, MLpot symbols, Packmol, and libcharmm freshness |
| `mmml doctor --checkpoint path.json` | verifying that a candidate ML checkpoint path resolves during setup | everything above plus the checkpoint path probe |
| `mmml health-check --require-gpu` | entering a GPU allocation before an MLpot run | the default interface probes, with CPU-only JAX treated as failure |
| `MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check --strict` | validating an MPI-linked CHARMM/MLpot launch shell | warnings become failures and the check runs through the MPI launcher |
| `MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check --live --checkpoint "$MMML_CKPT"` | final smoke before spending scheduler time | all default probes plus live MLpot registration and one CHARMM `ENER` on a tiny DCM cluster |

The default `health-check` sections are `core`, `jax`, `gpu_quantum`,
`charmm`, `mlpot`, `packmol`, `checkpoint`, and `mpi`. Use `--only` for a
single failing area while debugging, for example:

```bash
mmml health-check --only jax gpu_quantum --require-gpu
mmml health-check --only checkpoint --checkpoint "$MMML_CKPT" --json
mmml health-check --skip checkpoint mpi
```

Read the summary literally. A CPU-only JAX result is OK unless the command was
run with `--require-gpu`; the `gpu_quantum` section is skipped when `cupy` is not
installed; and the `mlpot` section checks libcharmm symbols, not an energy
evaluation. Add `--live` when you need to prove that PyCHARMM can register MLpot
and evaluate a tiny system.

Checkpoint checks are path and metadata checks, not model validation. The report
resolves `MMML_CKPT` or `--checkpoint`, peeks JSON keys for `.json` files, and
prints a short provenance hint such as "bundled HF portable", "DESdimers /
ACO-like organics", "peptide-oriented", or "SpookyNet example weights" based on
the path name. Treat that hint as a guardrail for obvious mixups; it does not
prove the weights fit your chemistry or that a production MD run will be stable.

## What's here

**How-to**

- [SciCORE cluster guide](../scicore.md) — partitions, modules, submission.
- [MPI operations](../pycharmm-mpi.md) and
  [Threading & launchers](../pycharmm-threading.md) — the two settings most
  often responsible for a slow run.
- [FFTW for CHARMM](../fftw-build.md) — building the dependency by hand.
- [Periodic boundaries (IMAGE super system)](../pbc-super-system.md) and
  [PyCHARMM C API (box and pressure)](../pycharmm-c-api-pbc-box-pressure.md) —
  the PBC and barostat plumbing.

**Scale and performance**

- [Calculator profiling](../calculator-profiling.md) — separating JAX compile
  time from run time before you optimise the wrong thing.
- [Medium PBC (500–2000 monomers)](../mlpot-medium-pbc.md) — what changes at
  that size.
- [Spatial ML MPI](../mlpot-spatial-mpi.md) — domain decomposition for the ML
  region.

**Commands** — `env`, `configure`, `doctor`, `completion`, `gui`,
`unwrap-traj`, and the plotting/diagnostic helpers.

## A warning about JIT

A first `md-system` step on a GPU node can spend minutes in XLA compilation, and
that time is easy to misread as a slow simulation. `mmml warmup-mlpot-jax` pays
it once, up front — see [MD & campaigns](md-campaigns.md).
