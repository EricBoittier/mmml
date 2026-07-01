# `mmml env`

Find resolved/bundled checkpoints and CHARMM paths.


Resolve checkpoints, CHARMM paths, and shell export hints without importing
PyCHARMM.

```bash
mmml env
mmml env --json
```

## Usage

```bash
mmml env --help
```

## Options

```text
usage: mmml env [-h] [--json] [--export]

Show resolved MMML paths (checkpoint, CHARMM, preset locations) and suggested
export lines for MMML_CKPT.

options:
  -h, --help  show this help message and exit
  --json      Print machine-readable JSON.
  --export    Print export lines only (for eval "$(mmml env --export)").
```

## Related docs

- [CLI overview](../index.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
