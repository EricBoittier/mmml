# `mmml md-embedding`

Solvated peptide partial MLpot (train/build/run).


## Usage

```bash
mmml md-embedding --help
```

## Options

```text
usage: mmml md-embedding [-h] {train,build,run} ...

Solvated-peptide MD embedding: train PhysNet on peptide NPZ, build CHARMM
PEPT+TIP3 box, register partial MLpot (n_monomers=1). See docs/examples/md-
embedding-design.md.

positional arguments:
  {train,build,run}
    train            Download/split aaa.ama NPZ, run PhysNet smoke, export JSON
                     checkpoint.
    build            Build CGENFF TRIA + TIP3 box; MM SD minimize; write
                     model.psf/crd/box.json.
    run              Load built box, register partial MLpot on PEPT, optional
                     MLpot SD.

options:
  -h, --help         show this help message and exit
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
