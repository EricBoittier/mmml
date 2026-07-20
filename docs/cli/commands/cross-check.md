# `mmml cross-check`

Supplementary QC cross-check.


## Usage

```bash
mmml cross-check --help
```

## Options

```text
usage: mmml cross-check [-h] [-c CONFIG] [-i STRUCTURES] [-o OUTPUT_DIR]
                        [--reference-npz REFERENCE_NPZ] [--reference REFERENCE]
                        [--backend BACKEND_NAMES] [--checkpoint CHECKPOINT]
                        [--functional FUNCTIONAL] [--basis BASIS]
                        [--max-frames MAX_FRAMES] [--stride STRIDE]
                        [--charge CHARGE] [--spin SPIN]
                        [--multiplicity MULTIPLICITY] [--no-plots]
                        [--no-save-backend-npz] [--orca-template ORCA_TEMPLATE]
                        [--molpro-template MOLPRO_TEMPLATE]

Supplementary QC cross-check (PySCF, ORCA QM, xTB, Molpro, ML).

Input & configuration:
  -c, --config CONFIG   YAML config file (see
                        examples/cross_check/cross_check.example.yaml)
  -i, --input, --structures STRUCTURES
                        Input NPZ or XYZ with structures to evaluate
  --checkpoint CHECKPOINT
                        ML checkpoint (shorthand for --backend ml)
  --orca-template ORCA_TEMPLATE
                        Custom ORCA input template with
                        {xyz},{method},{basis},{charge},{mult} placeholders
  --molpro-template MOLPRO_TEMPLATE
                        Custom Molpro input template with
                        {geometry},{basis},{method},{charge},{mult} placeholders

Scientific model:
  --functional, --xc FUNCTIONAL
                        XC functional for pyscf/orca reference backend
  --basis BASIS         Basis set for pyscf/orca/molpro backends
  --charge CHARGE       Total charge (default: 0)
  --spin SPIN           2*spin for PySCF (default: 0)

Execution:
  --backend BACKEND_NAMES
                        Backend to evaluate (repeatable): pyscf, ml, orca, xtb,
                        molpro

Output & artifacts:
  -o, --output-dir OUTPUT_DIR
                        Output directory (default: cross_check_out)
  --no-plots            Skip matplotlib comparison plots
  --no-save-backend-npz
                        Do not write per-backend NPZ files

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --reference-npz REFERENCE_NPZ
                        Use existing reference NPZ instead of running a
                        reference backend
  --reference REFERENCE
                        Reference backend name when --reference-npz is not set
                        (default: pyscf)
  --max-frames MAX_FRAMES
                        Maximum number of structures to evaluate
  --stride STRIDE       Frame stride (default: 1)
  --multiplicity MULTIPLICITY
                        Spin multiplicity for ORCA/Molpro/xTB (default: spin+1)

Run supplementary QC cross-checks against a reference (PySCF, ORCA QM, xTB,
Molpro, ML). Examples -------- From YAML config: mmml cross-check -c
cross_check.example.yaml CLI flags (minimal smoke): mmml cross-check -i
sampled.npz --reference-npz ref.npz \ --backend ml --checkpoint epoch.pkl -o
validation/ mmml cross-check -i water.xyz --reference pyscf --backend xtb --max-
frames 1
```


## Related docs

- [QC cross-check](../../qc-cross-check.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
