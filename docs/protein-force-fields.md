# Protein force fields — CHARMM and jax-md

Build small peptides with **CHARMM36 all-atom protein** parameters (PyCHARMM) and evaluate **JAX** bonded or full MM energies with **jax-md** or MMML loaders.

Related: [Tri-alanine water box](trialanine-water-box.md) (bundled CGENFF peptide), [CHARMM CGenFF JAX clone](cgenff-jax-clone.md), [Structure building](cli/structure-building.md).

---

## Force-field map

| Stack | jax-md module | Typical input | Protein-ready? |
|-------|---------------|---------------|----------------|
| **CHARMM all36 protein** | `oplsaa.load_charmm_system` | `top_all36_prot.rtf` + `par_all36m_prot.prm` + PDB | Yes (via CHARMM files) |
| **CHARMM CGENFF** | `io.charmm.parse_*` (via MMML) | bundled `top_all36_cgenff.rtf` + PSF | Small molecules + bundled `TRIA` peptide |
| **OPLS-AA native** | `oplsaa.create_topology` / `create_parameters` | Programmatic or CHARMM files | Same as CHARMM-file path |
| **AMBER** | `amber.energy` | Topology from **OpenMM** import (`openmm.py` tools) | Yes, after OpenMM conversion |
| **ReaxFF** | `reaxff` | Reactive FF; not standard fixed-charge protein MD | Specialized |

MMML production MLpot uses **CGENFF** for small-molecule liquids and **hybrid ML** for solutes; protein **CHARMM36** is supported for MM reference builds and jax-md cross-checks via the paths below.

---

## 1. CHARMM36 protein build (PyCHARMM)

Requires `CHARMM_HOME` with protein `toppar` (`top_all36_prot.rtf`, `par_all36m_prot.prm` or `par_all36_prot.prm`).

### Alanine dipeptide (ACE–ALA–CT3)

```bash
./scripts/mmml-charmm-mpirun.sh python scripts/examples/charmm_build_protein_alad.py \
  -o /tmp/alad_charmm
```

Writes:

```text
/tmp/alad_charmm/alad.pdb
/tmp/alad_charmm/alad.psf
```

Python API (`mmml.interfaces.pycharmmInterface.protein_charmm_build`):

```python
from pathlib import Path
from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

ensure_pycharmm_loaded()
from mmml.interfaces.pycharmmInterface.protein_charmm_build import (
    build_alad_dipeptide,
    protein_toppar_paths,
    write_alad_artifacts,
)

toppar = protein_toppar_paths()
print(toppar.rtf, toppar.prm)

pdb, psf, build = write_alad_artifacts("/tmp/alad_charmm", minimize=True)
print(build.n_atoms, pdb, psf)
```

### Longer peptides (sequence + patches)

Use standard PyCHARMM generation after loading protein toppar:

```python
from pycharmm import generate, ic, read, settings
from mmml.interfaces.pycharmmInterface.protein_charmm_build import protein_toppar_paths

toppar = protein_toppar_paths()
settings.set_verbosity(5)
read.rtf(str(toppar.rtf))
read.prm(str(toppar.prm))
read.sequence_string("ALA ALA ALA")  # four-char segment names in real workflows
generate.new_segment(
    seg_name="TRP1",
    first_patch="ACE",
    last_patch="CT3",
    setup_ic=True,
)
ic.prm_fill(replace_all=True)
ic.build()
```

For production solvated proteins, continue with [Packmol placement](packmol-placement.md) (`TIP3` waters) or `mmml liquid-box` after PSF/PDB export.

### MPI φ/ψ scan (workshop smoke)

```bash
MMML_MPI_NP=4 ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/charmm/mpi_alad_phi_psi.py --n-phi 12 --n-psi 12 \
  -o /tmp/alad_phi_psi_mpi.json
```

---

## 2. JAX evaluation (jax-md + MMML)

### MMML bonded loader (PSF + protein PRM)

Matches `cgenff_bonded.py` / `cgenff_topology.py` (jax-md CHARMM parsers, CMAP and Urey–Bradley when present in the PRM):

```bash
JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \
  --pdb /tmp/alad_charmm/alad.pdb \
  --psf /tmp/alad_charmm/alad.psf \
  --prm "$CHARMM_HOME/toppar/par_all36m_prot.prm" \
  --loader mmml-bonded
```

```python
import jax.numpy as jnp
import numpy as np
from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
from mmml.interfaces.pycharmmInterface.cgenff_topology import load_cgenff_bonded_from_psf
from mmml.interfaces.pycharmmInterface.protein_charmm_build import protein_toppar_paths

positions = np.loadtxt(...)  # or ASE read
toppar = protein_toppar_paths()
system = load_cgenff_bonded_from_psf(
    "/tmp/alad_charmm/alad.psf",
    positions,
    prm_file=toppar.prm,
)
energy, forces = bonded_energy_and_forces(
    jnp.asarray(positions),
    system.topology,
    system.bonded,
    energy_unit="kcal/mol",
)
```

### jax-md OPLS-AA (`load_charmm_system`)

Bonded-only (no neighbor list):

```bash
JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \
  --pdb /tmp/alad_charmm/alad.pdb \
  --rtf "$CHARMM_HOME/toppar/top_all36_prot.rtf" \
  --prm "$CHARMM_HOME/toppar/par_all36m_prot.prm" \
  --loader jaxmd-oplsaa
```

Bonded + cutoff nonbonded (vacuum box):

```bash
JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \
  --pdb /tmp/alad_charmm/alad.pdb \
  --rtf "$CHARMM_HOME/toppar/top_all36_prot.rtf" \
  --prm "$CHARMM_HOME/toppar/par_all36m_prot.prm" \
  --loader jaxmd-oplsaa --nonbonded --box-side 50
```

Native jax-md API:

```python
import jax.numpy as jnp
from jax_md.mm_forcefields.base import NonbondedOptions
from jax_md.mm_forcefields.nonbonded.electrostatics import CutoffCoulomb
from jax_md.mm_forcefields.oplsaa import energy, load_charmm_system

positions, topology, parameters = load_charmm_system(
    "alad.pdb",
    "par_all36m_prot.prm",
    "top_all36_prot.rtf",
)
box = jnp.array([50.0, 50.0, 50.0])
coulomb = CutoffCoulomb(r_cut=12.0)
nb = NonbondedOptions(r_cut=12.0, use_pbc=False)
energy_fn, nbr_fn, disp_fn, shift_fn = energy(topology, parameters, box, coulomb, nb)
nbrs = nbr_fn.allocate(positions)
E = energy_fn(positions, nbrs)
print({k: float(E[k]) for k in ("bond", "angle", "torsion", "vdw", "coulomb", "total")})
```

For periodic protein/water boxes, set `use_pbc=True`, pass the simulation cell as `box`, and prefer `PMECoulomb` or `EwaldCoulomb` over bare cutoff Coulomb.

### AMBER (OpenMM import)

jax-md's `amber.energy` expects systems converted from OpenMM (see jax-md `openmm.py` and `mm_forcefields/amber/energy.py` docstring). Typical workflow:

1. Build protein in OpenMM with `amber14-all.xml` (or similar).
2. Export CHARMM-like topology/parameters through jax-md's OpenMM conversion utilities.
3. Call `amber.energy(...)` with the resulting `Topology` + `Parameters`.

MMML does not ship OpenMM protein builders; use OpenMM directly for that path, then jax-md for JIT dynamics.

---

## 3. General Peptide Builder & Solvation (CGenFF + Protein append)

For arbitrary residue sequences, MMML provides a general peptide builder that supports terminal patching, solvation, and Quality Control (QC) structural checks:

```python
from mmml.interfaces.pycharmmInterface.peptide_builder import (
    build_peptide_in_charmm,
    solvate_peptide_in_charmm,
    qc_built_system,
    infer_charge_and_spin_from_psf,
)

# 1. Build arbitrary peptide sequence from 3-letter codes, space-separated, or 1-letter codes
peptide = build_peptide_in_charmm(
    "AAA",             # or "ALA ALA ALA" or ["ALA", "ALA", "ALA"]
    first_patch="ACE",  # default acetylated/neutral terminal patches
    last_patch="CT3",
)

# 2. Solvate in a cubic box with TIP3 water centered at (0, 0, 0) to avoid PBC wrapping
box = solvate_peptide_in_charmm(peptide, box_side_A=28.0, n_waters=100)

# 3. Perform structural QC (checks charges, bond lengths, and non-bonded steric clashes)
report = qc_built_system(box.positions, box.psf_path, check_energy=True)
if report.is_valid:
    print(f"System is valid! Energy: {report.details['charmm_energy']:.3f} kcal/mol")
```

The builder loads the CGenFF parameters first as the primary topology and appends the standard protein parameters second. This ensures that water and standard ion chemical types (`OT`, `HT`, `CLA`, `SOD`) are correctly mapped inside PyCHARMM.

Furthermore, `infer_charge_and_spin_from_psf` can be used to automatically determine the total charge and spin multiplicity of the constructed peptide system from the PSF:
```python
total_charge, spin_multiplicity = infer_charge_and_spin_from_psf(box.psf_path)
print(f"Charge: {total_charge}, Spin Multiplicity: {spin_multiplicity}")
```

---

## 4. Bundled CGENFF peptide (no protein toppar)

Tri-alanine in periodic water uses a supplemental CGENFF residue **`TRIA`** — no `top_all36_prot.rtf` at runtime:

```bash
./scripts/mmml-charmm-mpirun.sh python -c "
from pathlib import Path
from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
ensure_pycharmm_loaded()
from mmml.interfaces.pycharmmInterface.trialanine_water_box import build_trialanine_water_box_in_charmm
box = build_trialanine_water_box_in_charmm(n_waters=10, box_side_A=28.0, workdir=Path('/tmp/tria'))
print(box.n_atoms if hasattr(box, 'n_atoms') else len(box.positions), box.psf_path)
"
```

JAX cross-check: [trialanine-water-box.md](trialanine-water-box.md).

---

## 5. Choosing a path

| Goal | Recommendation |
|------|----------------|
| Standard protein MM reference | CHARMM36 toppar + PyCHARMM build |
| JAX bonded cross-check vs CHARMM | `load_cgenff_bonded_from_psf` + protein PRM |
| JAX full MM (vacuum/PBC) | `oplsaa.load_charmm_system` + `oplsaa.energy` |
| AMBER-family protein | OpenMM → jax-md `amber.energy` |
| MLpot liquid / small molecules | CGENFF + Packmol ([packmol-placement.md](packmol-placement.md)) |
| General peptide build and solvation | `build_peptide_in_charmm` + `solvate_peptide_in_charmm` |
| Peptide smoke without toppar | Bundled `TRIA` + TIP3 |

---

## Example scripts

| Script | Role |
|--------|------|
| `scripts/examples/charmm_build_protein_alad.py` | CHARMM36 ALAD → PDB/PSF |
| `scripts/examples/jaxmd_protein_alad_energy.py` | JAX energy from PDB (+ PSF/RTF/PRM) |
| `tests/functionality/charmm/mpi_alad_phi_psi.py` | MPI φ/ψ grid on ALAD |

User-run tests: [tests/functionality/protein/README.md](https://github.com/EricBoittier/mmml/blob/main/tests/functionality/protein/README.md).
