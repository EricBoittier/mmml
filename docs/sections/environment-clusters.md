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

## Preflight with `health-check`

Use [`mmml health-check`](../cli/commands/health-check.md) before a cluster run
when the question is "will this environment launch MLpot cleanly?" rather than
"is the current repository importable?". By default it runs fast probes for:

- core Python imports (`numpy`, `jax`, `e3x`, `ase`);
- JAX devices, with `--require-gpu` turning a CPU-only runtime into a failure;
- the `cupy`/`gpu4pyscf` combination used by GPU quantum Hessian paths;
- `libcharmm` and MLpot callback symbols;
- Packmol, checkpoint resolution, and OpenMPI / `mpi4py` compatibility.

The checkpoint probe resolves `--checkpoint` or `$MMML_CKPT`, reports the path,
and prints a short chemistry/provenance hint such as "DESdimers / ACO-like
organics" or "peptide-oriented". For portable JSON checkpoints it only peeks at
top-level keys; it does not instantiate the model or validate that the weights
fit your system chemistry.

For production MLpot jobs on MPI-linked CHARMM builds, run the same probe under
the repository launcher so the OpenMPI bootstrap matches the eventual MD job:

```bash
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check \
  --require-gpu --strict --checkpoint "$MMML_CKPT"
```

Add `--live` when you want a real MLpot registration plus CHARMM `ENER` smoke on
a tiny cluster. This is still a preflight, not an MD validation: it proves that
the checkpoint can be resolved, MLpot can register, and CHARMM can evaluate one
energy for the requested residue/count.

```bash
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh health-check \
  --live --live-residue DCM --live-n-molecules 2 --checkpoint "$MMML_CKPT"
```

Useful variants:

- `--json` emits the same report as structured data for CI or scheduler logs.
- `--strict` treats warnings as failures; combine it with `--prelaunch` for a
  serial preflight where "not under mpirun yet" should not fail the job.
- `--tier2` nests the spatial-MPI GPU checks from `mpi-check --tier2` inside the
  MPI section when validating `MMML_MLPOT_SPATIAL_MPI=1` workflows.
- `--only core jax checkpoint` or `--skip packmol` keeps login-node checks fast
  when CHARMM or Packmol is intentionally unavailable there.

Common interpretations:

| Symptom | Meaning | Next step |
| --- | --- | --- |
| `No JAX CUDA devices visible` with `--require-gpu` | The job is on CPU or CUDA runtime libraries are missing. | Load the cluster CUDA/cuDNN modules, request a GPU node, or install the GPU extra before running MD. |
| `cupy >= 14` warning with `gpu4pyscf` | GPU DFT Hessian/CPSCF can hit the known cuTENSOR fallback issue. | Pin a 13.x `cupy-cuda*` package for that workflow, or ignore it for CPU-only/non-Hessian jobs. |
| `Set MMML_CKPT or pass --checkpoint` | MLpot weights were not resolved. | Export `MMML_CKPT` or pass `--checkpoint` explicitly in the same launcher environment used for MD. |
| `MPI-linked libcharmm` warning outside `mpirun` | The library is importable, but production MLpot should use the wrapper. | Re-run with `./scripts/mmml-charmm-mpirun.sh`; see [MPI operations](../pycharmm-mpi.md). |

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

**Commands** — `env`, `configure`, `doctor`, `health-check`, `completion`,
`gui`, `unwrap-traj`, and the plotting/diagnostic helpers.

## A warning about JIT

A first `md-system` step on a GPU node can spend minutes in XLA compilation, and
that time is easy to misread as a slow simulation. `mmml warmup-mlpot-jax` pays
it once, up front — see [MD & campaigns](md-campaigns.md).
