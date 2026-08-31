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

## `health-check` preflight ladder

Use [`mmml health-check`](../cli/commands/health-check.md) immediately before
PyCHARMM / MLpot work on a new node, after loading modules, or after changing a
checkpoint. It is a preflight, not a simulation: the default probes cover Python
imports, JAX devices, the GPU-DFT `cupy` / `gpu4pyscf` footgun, `libcharmm`,
MLpot symbols, Packmol, checkpoint path resolution, and MPI launch state. A
passing default report means the interface pieces are visible; it does not prove
that a full MD stage will conserve energy or that a checkpoint fits the target
chemistry.

```bash
# Fast login-node or batch-prologue check. Fails if JAX cannot see CUDA.
mmml health-check --require-gpu

# Machine-readable record for CI, Slurm logs, or bug reports.
mmml health-check --require-gpu --json > health-check.json

# Treat warnings as blocking in scripted gates.
mmml health-check --require-gpu --strict
```

Add `--live` only on a node where CHARMM can run. It registers MLpot and asks
CHARMM for one `ENER` evaluation on a small residue cluster (`DCM:2` by
default), then exits; it still is not a minimization or dynamics smoke.

```bash
export MMML_CKPT=/path/to/params.json
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check \
  --require-gpu --live --checkpoint "$MMML_CKPT"

# Use a different residue if it better matches the checkpoint chemistry.
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check \
  --live --live-residue ACO --live-n-molecules 2 --checkpoint "$MMML_CKPT"
```

Interpret the checkpoint line conservatively. The command resolves
`--checkpoint` or `MMML_CKPT`, peeks at portable JSON keys, and prints a short
path-based hint such as "DESdimers / ACO-like organics" or "SpookyNet example
weights"; it does not instantiate the model unless `--live` runs, and it cannot
certify chemistry coverage. Follow the checkpoint checks in
[Training & sampling](training-sampling.md#before-you-trust-a-checkpoint),
`mmml mode-check`, or the relevant functionality workflow before production MD.

For MPI-linked CHARMM builds, run the live check through
`./scripts/mmml-charmm-mpirun.sh` even with `MMML_MPI_NP=1`. Use
`--tier2` when validating spatial-ML MPI (`MMML_MLPOT_SPATIAL_MPI=1`,
`--ml-spatial-mpi`); combine it with `--prelaunch --strict` for a serial
preflight that should ignore only the expected "not yet under mpirun" warning.

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
