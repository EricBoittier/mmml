# `mmml md-embedding`

Solvated-peptide partial MLpot workflow (train → build → run), separate from homogeneous
cluster [`md-system`](md-system.md).

## Usage

```bash
mmml md-embedding --help
mmml md-embedding train --help
mmml md-embedding build --help
mmml md-embedding run --help
```

## Phases

| Phase | CHARMM | Purpose |
|-------|--------|---------|
| `train` | No | Download, **`mmml fix-and-split`** (default), PhysNet smoke, JSON checkpoint |
| `build` | Yes | PEPT + TIP3 box, MM SD minimize, `model.psf` / `box.json` + ASE figures |
| `run` | Yes | Partial MLpot on `PEPT`, optional MLpot SD |

## Examples

```bash
mmml md-embedding train -o artifacts/md_embedding/aaa
mmml md-embedding build -o artifacts/md_embedding/aaa --n-waters 10
mmml md-embedding run -o artifacts/md_embedding/aaa \
  --checkpoint artifacts/md_embedding/aaa/aaa_smoke_params.json --mini-nstep 20
```

## Related docs

- [MD embedding design](../../examples/md-embedding-design.md)
- [aaa.ama peptide workflow](../../examples/aaa-ama-workflow.md)
- [Functionality smoke steps](../../../tests/functionality/embedding/README.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
