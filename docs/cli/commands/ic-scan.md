# `mmml ic-scan`

Bond/angle/dihedral scans (1D or N-D) for QM/ML.


## Usage

```bash
mmml ic-scan --help
```

## Options

```text
usage: mmml ic-scan [-h] --config CONFIG [--prepare-only] [--allow-partial]
                    [--overwrite] --output OUTPUT

Prepare and optionally evaluate bond/angle/dihedral scans from a config that
defines DoFs, grids, and 1D or N-D scan combinations.

Input & configuration:
  --config CONFIG  YAML/JSON IcScanConfig (structure, dofs, scan_mode/scans,
                   calculator)

Output & artifacts:
  --overwrite
  --output OUTPUT

Diagnostics & safety:
  -h, --help       show this help message and exit
  --allow-partial  Exit 0 even if some energy evaluations fail

Other options:
  --prepare-only   Write geometries without energy evaluation (overrides
                   evaluate)
```

## Visual examples

![Trialanine PES with force-annotated conformers](../../images/povray-overlays/trialanine_pes_with_povray.png)

## Related docs

- [Internal-coordinate scan design](../../ic-scan-design.md)
- [Scientific code policy](../../scientific-code.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
