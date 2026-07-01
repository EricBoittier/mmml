# `mmml build-crystal`

Symmetry-aware crystals (PyXtal).


Build molecular crystals with PyXtal (`uv sync --extra chem`). For **DCM**
(CH₂Cl₂), prefer the deposited experimental structure
([CCDC doi:10.5517/cc9lyjb](https://www.ccdc.cam.ac.uk/structures/search?id=doi:10.5517/cc9lyjb&sid=DataCite);
COD [2100015](https://www.crystallography.net/2100015.html)) — **Pbcn**, Z=4,
ρ≈**1.97 g/cm³** at 1.63 GPa / 293 K. PyXtal cannot build from `C(Cl)Cl` SMILES;
use the bundled monomer XYZ or CIF.

```bash
# Experimental crystal → extxyz / NPZ handoff (no PyXtal)
python -c "
from ase.io import read, write
from mmml.paths import default_dcm_crystal_cif
atoms = read(default_dcm_crystal_cif())
write('dcm_expt.extxyz', atoms)
"

# PyXtal random placement in the same space group (SG 60 = Pbcn)
mmml build-crystal \
  -m "$(python -c 'from mmml.paths import default_dcm_molecule_xyz; print(default_dcm_molecule_xyz())')" \
  --spg 60 --z 4 \
  --target-density-g-cm3 1.972 \
  -o dcm_pyxtal.extxyz

# Benzene (SMILES in PyXtal DB)
mmml build-crystal -m c1ccccc1 --spg 14 --z 2 -o benzene.extxyz
```

Liquid DCM boxes use **1.326 g/cm³** (`liquid-box`, `md-system`).

## Usage

```bash
mmml build-crystal --help
```

## Options

```text
usage: mmml build-crystal [-h] -m SPEC [--stoichiometry Z [Z ...]]
                          [--z Z_VALUES [Z_VALUES ...]] [--dim {0,1,2,3}]
                          [--spg SPACE_GROUP] [--factor FACTOR]
                          [--target-density-g-cm3 RHO] [--seed SEED]
                          [--attempts ATTEMPTS] [--no-resort]
                          [--supercell NX,NY,NZ] -o OUTPUT
                          [--format OUT_FORMAT] [--optimize]
                          [--optimizer {bfgs,fire,lbfgs}] [--fmax FMAX]
                          [--max-opt-steps MAX_OPT_STEPS] [--fix-cell] [--emt]
                          [--quiet-opt]

Build molecular crystals with PyXtal (space-group symmetry) and export ASE-
compatible structures for optimization or MMML handoff.

options:
  -h, --help            show this help message and exit
  -m, --molecule SPEC   Molecule specification (repeat for multi-component
                        crystals): XYZ/CIF path, SMILES, or chemical formula
                        understood by PyXtal (default: None)
  --stoichiometry Z [Z ...]
                        Formula units per molecule species (same order as
                        --molecule) (default: None)
  --z Z_VALUES [Z_VALUES ...]
                        Alias for stoichiometry; one value repeats for all
                        molecules (default: None)
  --dim {0,1,2,3}       Crystal dimensionality (0=cluster, 3=3D periodic)
                        (default: 3)
  --spg, --space-group SPACE_GROUP
                        International space-group number (default: 14)
  --factor FACTOR       PyXtal volume factor passed to from_random (default:
                        1.0)
  --target-density-g-cm3 RHO
                        Scale unit cell after build to this mass density
                        (g/cm³). Liquid DCM ≈ 1.326; experimental Pbcn crystal
                        (COD 2100015) ≈ 1.97 (default: None)
  --seed SEED           RNG seed for reproducible PyXtal trials (default:
                        None)
  --attempts ATTEMPTS   Maximum PyXtal from_random retries (default: 20)
  --no-resort           Keep PyXtal atom order in ASE export (to_ase
                        resort=False) (default: False)
  --supercell NX,NY,NZ  Build supercell after generation (e.g. 2,2,2)
                        (default: None)
  -o, --output OUTPUT   Output path (.xyz, .extxyz, .cif, or .npz) (default:
                        None)
  --format OUT_FORMAT   ASE output format override (default: inferred from
                        --output suffix) (default: None)

ASE optimization (optional):
  --optimize            Relax structure with ASE after PyXtal generation
                        (default: False)
  --optimizer {bfgs,fire,lbfgs}
                        ASE optimizer when --optimize is set (default: bfgs)
  --fmax FMAX           ASE force convergence (eV/Å) (default: 0.05)
  --max-opt-steps MAX_OPT_STEPS
                        Maximum ASE optimizer steps (default: 200)
  --fix-cell            Document intent to keep the unit cell fixed
                        (positions-only relaxation) (default: False)
  --emt                 Use ASE EMT calculator for --optimize (smoke tests
                        only) (default: False)
  --quiet-opt           Suppress ASE optimizer log output (default: False)
```

## Example structures

![DCM crystal / periodic cell (experimental Pbcn)](../../images/structures/build-crystal.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Structure building guide](../structure-building.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
