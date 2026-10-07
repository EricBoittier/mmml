# `karml env`

Find resolved/bundled checkpoints and CHARMM paths.


Resolve checkpoints, CHARMM paths, and shell export hints without importing
PyCHARMM.

```bash
karml env
karml env --json
```

## Usage

```bash
karml env --help
```

## Options

```text
usage: karml env [-h] [--json] [--export]

Show resolved KARML environment paths and status of default model parameters
(PhysNet, SpookyNet, MBD, Multipoles).

options:
  -h, --help  show this help message and exit
  --json      Print machine-readable JSON.
  --export    Print export lines only (for eval "$(karml env --export)").
```


## Related docs

- [CLI overview](../index.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
