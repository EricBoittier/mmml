# `mmml mpi-launch`

Launch OpenMPI with an explicit JAX execution policy.


## Usage

```bash
mmml mpi-launch --help
```

## Options

```text
usage: mmml mpi-launch [-h] [--mpi-ranks MPI_RANKS]
                       [--jax-mode {cpu-threaded,gpu-single,gpu-per-rank,rank0,spatial}]
                       [--jax-cpu-threads JAX_CPU_THREADS]
                       [--charmm-omp-threads CHARMM_OMP_THREADS]
                       [--preset {single,cpu,spatial}] [--strict-resources]
                       [--dry-run]
                       ...

Launch CHARMM/OpenMPI with an independent JAX device/thread policy. When invoked
through 'uv run', the active uv interpreter is used on every rank.

positional arguments:
  command

options:
  -h, --help            show this help message and exit
  --mpi-ranks MPI_RANKS
  --jax-mode {cpu-threaded,gpu-single,gpu-per-rank,rank0,spatial}
  --jax-cpu-threads JAX_CPU_THREADS
  --charmm-omp-threads CHARMM_OMP_THREADS
  --preset {single,cpu,spatial}
  --strict-resources
  --dry-run
```


## Related docs

- [PyCHARMM MPI](../../pycharmm-mpi.md)
- [Spatial ML MPI](../../mlpot-spatial-mpi.md)
- [PyCHARMM threading](../../pycharmm-threading.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
