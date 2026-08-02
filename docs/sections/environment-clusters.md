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
