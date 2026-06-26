# PyXtal structure building (optional)

PyXtal integration lives behind the **`chem`** optional extra:

```bash
uv sync --extra chem
```

## CLI: `mmml build-crystal`

Generate a symmetry-aware molecular crystal and export ASE-compatible geometry:

```bash
# Random molecular crystal (space group 14, Z=2 for one species)
mmml build-crystal -m benzene.xyz --spg 14 --z 2 -o crystal.extxyz

# Co-crystal (two molecule types)
mmml build-crystal -m donor.xyz -m acceptor.xyz --stoichiometry 1 1 --spg 1 -o cocrystal.cif

# Supercell + quick EMT relaxation (smoke test)
mmml build-crystal -m h2o.xyz --spg 36 --z 4 --supercell 2,2,2 \
  --optimize --emt --optimizer fire -o h2o_relaxed.extxyz

# NPZ for md-system / handoff workflows
mmml build-crystal -m monomer.xyz --spg 4 --z 2 -o seed.npz
```

## Python API

```python
from mmml.interfaces.pyxtal_placement import (
    MolecularCrystalBuildRequest,
    build_molecular_crystal_random,
)
from mmml.interfaces.aseInterface.pyxtal_optimize import optimize_ase_atoms

req = MolecularCrystalBuildRequest(
    molecules=["benzene.xyz"],
    stoichiometry=[2],
    space_group=14,
    seed=42,
)
result = build_molecular_crystal_random(req)
atoms = result.atoms  # ase.Atoms with cell + PBC

# Attach your MMML / CHARMM ASE calculator, then relax:
# atoms.calc = my_hybrid_calc
# opt = optimize_ase_atoms(atoms, fix_cell=True, fmax_ev_a=0.05)
```

## MMML workflow notes

| Step | Tool |
|------|------|
| Symmetry-aware crystal generation | PyXtal (`build-crystal` or API) |
| Quick gas-phase / unit-cell test opt | ASE + EMT (`--optimize --emt`) |
| Production relaxation | ASE + MMML hybrid calculator (`calculator_minimize.py` patterns) |
| CHARMM topology + MD | `mmml make-res` per species, then `md-system` with NPZ/PDB handoff |
| Disordered liquid boxes | Packmol (`md-system --packmol`) — not PyXtal |

Existing research code in `mmml/generate/dimers.py` also uses PyXtal for symmetry-scanned dimers; the new `pyxtal_placement` module is the supported path for crystal → ASE export.

## Tests

```bash
pytest tests/unit/test_pyxtal_placement.py -q
```

Live PyXtal smoke (requires `mmml[chem]`):

```bash
mmml build-crystal -m O -m H -m H --stoichiometry 1 2 2 --spg 36 --z 4 --attempts 30 -o /tmp/h2o.extxyz
```
