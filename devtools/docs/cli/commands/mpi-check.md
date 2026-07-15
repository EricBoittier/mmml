# `mmml mpi-check`

Validate OpenMPI/CHARMM/mpi4py for MLpot.


## Usage

```bash
mmml mpi-check --help
```

## Options

```text
usage: mmml mpi-check [-h] [--json] [--strict] [--prelaunch] [--tier2] [--tier3]

Validate OpenMPI / CHARMM / mpi4py setup for PyCHARMM MLpot runs.

options:
  -h, --help   show this help message and exit
  --json       Emit machine-readable JSON on stdout.
  --strict     Treat warnings as errors (non-zero exit).
  --prelaunch  Serial pre-flight before mpirun: relax strict Tier 2 warnings
               (spatial env unset, np=1).
  --tier2      Also validate Tier 2 spatial MPI + GPU environment for MLpot.
  --tier3      Survey Tier 3 DOMDEC + MLpot blockers (informational; production
               blocked).
```


## Related docs

- [PyCHARMM MPI](../../pycharmm-mpi.md)
- [Spatial ML MPI](../../mlpot-spatial-mpi.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
