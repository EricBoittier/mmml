# `mmml run-pycharmm`

Pure CHARMM heating/equilibration.

!!! warning "legacy"
    Legacy command. Prefer **`mmml md-system --backend pycharmm (no ML checkpoint)`**. Pure MM CHARMM without MLpot; md-system covers ML workflows.


## Usage

```bash
mmml run-pycharmm --help
```

## Options

```text
usage: mmml run-pycharmm [-h] --pdbfile PDBFILE [--cell CELL]
                         [--skip-setup-energy-show]
                         [--pycharmm-minimize/--no-pycharmm-minimize | --no-pycharmm-minimize/--no-pycharmm-minimize]
                         [--pycharmm-minimize-steps N]
                         [--two-residue-sampling | --no-two-residue-sampling]
                         [--two-residue-restraint-force K]
                         [--two-residue-restraint-r0 ANGSTROM]
                         [--two-residue-sampling-steps N]
                         [--two-residue-restraint-resid1 TWO_RESIDUE_RESTRAINT_RESID1]
                         [--two-residue-restraint-resid2 TWO_RESIDUE_RESTRAINT_RESID2]
                         [--view-braille]

Pure PyCHARMM: heating and equilibration only (no MM/ML)

options:
  -h, --help            show this help message and exit
  --pdbfile PDBFILE     Path to the PDB file (requires correct atom names and
                        types for CHARMM).
  --cell CELL           Cubic cell side length in Å for periodic boundary
                        conditions (default: 1000).
  --skip-setup-energy-show
                        Skip energy.show() in setup_box for faster startup.
  --pycharmm-minimize/--no-pycharmm-minimize, --no-pycharmm-minimize/--no-pycharmm-minimize
                        Run PyCHARMM nbonds/minimize before heat (default:
                        True).
  --pycharmm-minimize-steps N
                        ABNR minimization steps (default: 1000).
  --two-residue-sampling, --no-two-residue-sampling
                        Run restrained two-residue PyCHARMM sampling after
                        nbonds/block setup (default: True).
  --two-residue-restraint-force K
                        CHARMM harmonic restraint force constant for two-residue
                        sampling (default: 1.0).
  --two-residue-restraint-r0 ANGSTROM
                        CHARMM harmonic restraint target distance r0 for two-
                        residue sampling (default: 2.5 Angstrom).
  --two-residue-sampling-steps N
                        ABNR steps for restrained two-residue sampling (default:
                        --pycharmm-minimize-steps).
  --two-residue-restraint-resid1 TWO_RESIDUE_RESTRAINT_RESID1
                        First CHARMM residue id for two-residue sampling
                        (default: 1).
  --two-residue-restraint-resid2 TWO_RESIDUE_RESTRAINT_RESID2
                        Second CHARMM residue id for two-residue sampling
                        (default: 2).
  --view-braille        Display braille molecular viewer at each phase.

Pure PyCHARMM runner: heating and equilibration only (no MM/ML). Runs CHARMM
setup, minimization, heating, and equilibration. Does not run ASE MD, JAX-MD, or
any ML calculator. Use this for classical CHARMM-only simulations or to prepare
structures before running mmml run (MM/ML). Usage: python -m
mmml.cli.run.run_pycharmm --pdbfile pdb/init-packmol.pdb --cell 40 mmml run-
pycharmm --pdbfile pdb/init-packmol.pdb --cell 40
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
