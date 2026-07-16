# `mmml doctor`

Is this machine ready? (JAX, CHARMM, Packmol).


## Usage

```bash
mmml doctor --help
```

## Options

```text
usage: mmml doctor [-h] [--json] [--require-gpu] [--mpi]
                   [--checkpoint CHECKPOINT] [--strict]

Check that this machine can run MMML (Python, JAX, CHARMM, Packmol).

options:
  -h, --help            show this help message and exit
  --json                machine-readable report on stdout
  --require-gpu         fail unless JAX sees a GPU
  --mpi                 also check OpenMPI / mpi4py wiring
  --checkpoint CHECKPOINT
                        also validate an ML checkpoint
  --strict              treat warnings as failures
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
