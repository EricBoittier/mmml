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

Input & configuration:
  --checkpoint CHECKPOINT
                        also validate an ML checkpoint

Execution:
  --mpi                 also check OpenMPI / mpi4py wiring

Output & artifacts:
  --json                machine-readable report on stdout

Diagnostics & safety:
  -h, --help            show this help message and exit
  --strict              treat warnings as failures

Other options:
  --require-gpu         fail unless JAX sees a GPU
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
