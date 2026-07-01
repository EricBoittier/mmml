# `mmml build-crystal`

Symmetry-aware crystals (PyXtal).


Build molecular crystals for MD. **Recommended for DCM and benzene:** literature
CIF + `make-res` atom names (`--literature dcm|benz`) — exact experimental unit
cell, tiled to a simulation supercell (≥28 Å edges by default) at literature ρ.

```bash
mmml make-res --res DCM --skip-energy-show
mmml build-crystal --literature dcm --monomer-pdb pdb/dcm.pdb -o pdb/dcm_crystal.pdb
mmml build-crystal --literature dcm --supercell 4,4,3 -o dcm_super.extxyz
```

PyXtal (`uv sync --extra chem`) is optional for random placement in the same
space group. DCM crystal: [COD 2100015](https://www.crystallography.net/2100015.html)
(Pbcn, ρ≈1.97 g/cm³). Benzene: [COD 4501704](https://www.crystallography.net/cod/4501704.html)
(P2₁/c, ρ≈1.20 g/cm³).

```bash
mmml build-crystal \
  -m "$(python -c 'from mmml.paths import default_dcm_molecule_xyz; print(default_dcm_molecule_xyz())')" \
  --spg 60 --z 4 --target-density-g-cm3 1.972 -o dcm_pyxtal.extxyz
mmml build-crystal -m benzene --spg 14 --z 2 --target-density-g-cm3 1.202 -o benzene.extxyz
```

Liquid DCM boxes use **1.326 g/cm³** (`liquid-box`, `md-system`).

Literature vs make-res+CIF vs PyXtal tables are in the
[structure building guide](../structure-building.md#literature-cross-check-auto-generated).

## Usage

```bash
mmml build-crystal --help
```

## Options

```text
usage: mmml build-crystal [-h] [--literature PRESET] [--from-cif PATH]
                          [--residue NAME] [--monomer-pdb PATH]
                          [--min-box-side ANG] [-m SPEC]
                          [--stoichiometry Z [Z ...]]
                          [--z Z_VALUES [Z_VALUES ...]] [--dim {0,1,2,3}]
                          [--spg SPACE_GROUP] [--factor FACTOR]
                          [--target-density-g-cm3 RHO] [--seed SEED]
                          [--attempts ATTEMPTS] [--no-resort]
                          [--supercell NX,NY,NZ] -o OUTPUT [--format OUT_FORMAT]
                          [--optimize] [--optimizer {bfgs,fire,lbfgs}]
                          [--fmax FMAX] [--max-opt-steps MAX_OPT_STEPS]
                          [--fix-cell] [--emt] [--quiet-opt]

Build molecular crystals: literature CIF + make-res (CHARMM names) or PyXtal
random placement with space-group symmetry.

options:
  -h, --help            show this help message and exit
  --target-density-g-cm3 RHO
                        Scale cell to this mass density (g/cm³). Literature
                        presets use CIF ρ unless this is set. Liquid DCM ≈
                        1.326; crystal DCM ≈ 1.972
  --supercell NX,NY,NZ  Supercell repeats (literature: auto from --min-box-side
                        if omitted)
  -o, --output OUTPUT   Output path (.pdb, .xyz, .extxyz, .cif, or .npz)
  --format OUT_FORMAT   ASE output format override (default: inferred from
                        --output suffix)

Literature CIF + make-res (recommended for DCM / benzene):
  --literature PRESET   Bundled experimental CIF preset: dcm (Pbcn) or benz
                        (P2₁/c)
  --from-cif PATH       Override CIF path (requires --residue or --literature
                        for residue name)
  --residue NAME        CHARMM residue (DCM, BENZ) when using --from-cif without
                        --literature
  --monomer-pdb PATH    make-res monomer PDB for atom-name mapping (default:
                        pdb/<res>.pdb or bundled)
  --min-box-side ANG    Minimum supercell edge length (Å); default ≈2× CHARMM
                        cutnb

PyXtal random placement:
  -m, --molecule SPEC   Molecule specification (repeat for multi-component
                        crystals): XYZ/CIF path, SMILES, or chemical formula
                        understood by PyXtal
  --stoichiometry Z [Z ...]
                        Formula units per molecule species (same order as
                        --molecule)
  --z Z_VALUES [Z_VALUES ...]
                        Alias for stoichiometry; one value repeats for all
                        molecules
  --dim {0,1,2,3}       Crystal dimensionality (0=cluster, 3=3D periodic)
  --spg, --space-group SPACE_GROUP
                        International space-group number
  --factor FACTOR       PyXtal volume factor passed to from_random
  --seed SEED           RNG seed for reproducible PyXtal trials
  --attempts ATTEMPTS   Maximum PyXtal from_random retries
  --no-resort           Keep PyXtal atom order in ASE export (to_ase
                        resort=False)

ASE optimization (optional, PyXtal path):
  --optimize            Relax structure with ASE after PyXtal generation
  --optimizer {bfgs,fire,lbfgs}
                        ASE optimizer when --optimize is set
  --fmax FMAX           ASE force convergence (eV/Å)
  --max-opt-steps MAX_OPT_STEPS
                        Maximum ASE optimizer steps
  --fix-cell            Document intent to keep the unit cell fixed (positions-
                        only relaxation)
  --emt                 Use ASE EMT calculator for --optimize (smoke tests only)
  --quiet-opt           Suppress ASE optimizer log output
```

## Example structures

![DCM crystal / periodic cell (experimental Pbcn)](../../images/structures/build-crystal.png)

More detail: [Structure building guide](../structure-building.md).

## Related docs

- [Structure building guide](../structure-building.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
