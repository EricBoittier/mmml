# `mmml pycharmm-two-residue-sample`

Restrained sampling for two-residue CHARMM system.


## Usage

```bash
mmml pycharmm-two-residue-sample --help
```

## Options

```text
usage: mmml pycharmm-two-residue-sample [-h] --pdbfile PDBFILE [--cell CELL]
                                        [--skip-setup-energy-show]
                                        [--pycharmm-minimize-steps N]
                                        [--output-pdb OUTPUT_PDB]
                                        [--two-residue-restraint-force K]
                                        [--two-residue-restraint-r0 ANGSTROM]
                                        [--two-residue-sampling-steps N]
                                        [--two-residue-restraint-resid1 TWO_RESIDUE_RESTRAINT_RESID1]
                                        [--two-residue-restraint-resid2 TWO_RESIDUE_RESTRAINT_RESID2]

Set up a two-residue CHARMM system and run restrained sampling.

options:
  -h, --help            show this help message and exit
  --pdbfile PDBFILE     Path to a PDB file containing the two-residue CHARMM
                        system.
  --cell CELL           Cubic cell side length in Angstrom for periodic
                        boundary conditions (default: 1000).
  --skip-setup-energy-show
                        Skip energy.show() in setup_box for faster startup.
  --pycharmm-minimize-steps N
                        Fallback ABNR steps when --two-residue-sampling-steps
                        is not set (default: 1000).
  --output-pdb OUTPUT_PDB
                        Path for sampled coordinates as PDB (default: pdb/two-
                        residue-sampled.pdb).
  --two-residue-restraint-force K
                        CHARMM harmonic restraint force constant for two-
                        residue sampling (default: 1.0).
  --two-residue-restraint-r0 ANGSTROM
                        CHARMM harmonic restraint target distance r0 for two-
                        residue sampling (default: 2.5 Angstrom).
  --two-residue-sampling-steps N
                        ABNR steps for restrained two-residue sampling
                        (default: --pycharmm-minimize-steps).
  --two-residue-restraint-resid1 TWO_RESIDUE_RESTRAINT_RESID1
                        First CHARMM residue id for two-residue sampling
                        (default: 1).
  --two-residue-restraint-resid2 TWO_RESIDUE_RESTRAINT_RESID2
                        Second CHARMM residue id for two-residue sampling
                        (default: 2).
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
