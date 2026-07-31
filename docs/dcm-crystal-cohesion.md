# Solid dichloromethane: halogen contacts, pressure and sublimation

Tests a published claim about which intermolecular contact holds crystalline
CH₂Cl₂ together, then relaxes the crystal to ambient pressure so its lattice
energy can honestly be compared with a sublimation enthalpy. CPU only, under a
minute — no CHARMM, no GPU, no trained checkpoint, no MD.

```bash
bash examples/dcm_crystal/run_all.sh
```

Reference: M. Podsiadło, K. F. Dziubek and A. Katrusiak, *In situ high-pressure
crystallization and compression of halogen contacts in dichloromethane*,
**Acta Crystallogr. B** 61, 595 (2005)
([doi:10.1107/S0108768105017374](https://doi.org/10.1107/S0108768105017374)),
CCDC [doi:10.5517/cc9lyjb](https://doi.org/10.5517/cc9lyjb). Coordinates via the
Crystallography Open Database.

## The bundled structures

The two entries below are the *only* pure CH₂Cl₂ structures in COD, and both come
from the paper above:

| `DCM_PHASE` | COD | Space group | a, b, c (Å) | Conditions | Z |
|---|---|---|---|---|---|
| `pbcn_133gpa` | 2100014 | Pbcn | 3.984, 7.863, 9.357 | 293 K, 1.33 GPa | 4 |
| `pbcn_163gpa` | 2100015 | Pbcn | 3.924, 7.793, 9.335 | 293 K, 1.63 GPa | 4 |

```python
from mmml.analysis.dcm_crystal import read_dcm_phase

atoms = read_dcm_phase("pbcn_133gpa", rebuild_hydrogens=True)  # 20 atoms
```

`default_dcm_crystal_cif()` still returns the 1.63 GPa structure, which is what
the `dcm` [`build-crystal`](cli/commands/build-crystal.md) preset and the
literature cross-check table have always used.

## The structure that is missing, and why it matters

Both deposited structures were grown in a diamond-anvil cell, and both are
compressed — by 10.7% and 13.0% relative to the ambient-pressure cell. **A
crystal squeezed 11% below its ambient volume sits well up its repulsive wall**,
so its static lattice energy is not a cohesive energy and comparing it directly
with a sublimation enthalpy is a category error.

The ambient-pressure phase is isostructural (Pbcn, Z = 4) and was determined by
Kawaguchi, Tanaka, Takeuchi and Watanabé, *Bull. Chem. Soc. Jpn.* **46**, 62
(1973) ([doi:10.1246/bcsj.46.62](https://doi.org/10.1246/bcsj.46.62)). It predates
CIF deposition and has no openly licensed coordinates, so MMML records only its
cell:

```python
from mmml.analysis.dcm_crystal import KAWAGUCHI_AMBIENT_CELL

KAWAGUCHI_AMBIENT_CELL.cell_lengths_A   # (4.249, 8.138, 9.492) at ~153 K
```

It is deliberately a different type from `DcmPhase` and has no `cif_path()`: it
is a target to relax *towards*, not a structure to build *from*.

## Rebuilding the hydrogens

X-rays scatter from electrons, and a hydrogen has a single bonding electron
displaced towards its heavy-atom partner. The consequences here are not subtle:
the two refinements put C–H at **1.01(10)** and **1.13(12) Å** and disagree on the
hydrogen *direction* by a comparable amount. Both spreads exceed the compression
between the two structures, and taken at face value they make the shortest H···Cl
contact appear to **lengthen** under pressure.

Normalising the C–H *distance*
([`normalize_hydrogen_positions`](#api)) is the usual crystallographic fix and it
is not enough here, because the directions disagree too. Since CH₂Cl₂ is C₂ᵥ and
its carbon and chlorines are located to a few thousandths of an Ångström, the
hydrogens follow from the heavy-atom frame plus two spectroscopic constants:

```python
from mmml.analysis.dcm_crystal import rebuild_methylene_hydrogens

fixed = rebuild_methylene_hydrogens(atoms)   # C-H 1.087 A, H-C-H 112 deg
```

| Shortest H···Cl | 1.33 GPa | 1.63 GPa | Change |
|---|---|---|---|
| deposited hydrogens | 2.771 Å | 2.888 Å | **+0.117** (wrong sign) |
| rebuilt hydrogens | 2.932 Å | 2.897 Å | −0.035 |

The rebuild also moves the CGenFF lattice energy by about 0.2 kcal/mol, so it is
a precondition for the energetics and not only for the geometry.

## Testing the paper's conclusion

The paper closes with a claim about *energy*:

> the crystal cohesion forces are dominated by H···Cl interactions rather than by
> Cl···Cl attractions

Diffraction cannot measure that directly, so the authors argued it from contact
compression and crystal habit. A force field can measure it:

```python
from mmml.analysis.lattice_energy import decompose_lattice_energy_by_element_pair

dec = decompose_lattice_energy_by_element_pair(
    fixed.get_positions(), fixed.get_atomic_numbers(), fixed.cell.array, cutoff_A=12.0
)
dec.dominant_contact()   # ('Cl', 'H')
```

At 1.33 GPa, in kcal/mol per molecule:

| Contact | Dispersion | Electrostatic | Total | Share |
|---|---|---|---|---|
| Cl···H | −4.257 | −1.233 | **−5.490** | 63% |
| H···H | −2.222 | +0.403 | −1.818 | 21% |
| Cl···Cl | −1.585 | **+0.148** | −1.436 | 16% |

CGenFF agrees with the paper, and the split shows why. The shortest Cl···Cl
contact is 3.358 Å — inside the 3.50 Å van der Waals sum — with a Type II
σ-hole geometry by the Desiraju–Parthasarathy criteria, so *geometrically* it is
a halogen bond. Yet its electrostatic contribution is repulsive and all of its
binding is dispersion. CGenFF puts a single point charge on chlorine and so
cannot produce an attractive halogen bond even in principle; what it can say is
that you do not need one, because the contact is bound by dispersion regardless.
A later plane-wave DFT study that *does* describe the σ-hole reached the same
conclusion — Kurzydłowski, Chumak and Rogoża, *Crystals* **10**, 920 (2020)
([doi:10.3390/cryst10100920](https://doi.org/10.3390/cryst10100920)) — finding
halogen bonds play "only a minor role" in CH₂Cl₂.

### Why molecule pairs and not atom pairs

The decomposition is over molecule pairs, each labelled by its shortest contact.
Each CH₂Cl₂ is neutral, so a molecule-pair energy is a well-defined interaction
energy. Splitting the same lattice by *atom* pair instead produces buckets of
order ±100 kcal/mol against a total near −9: the monopole terms cancel between
element pairs and mean nothing individually. A group-based cutoff is used for the
same reason — truncating atom pairs inside a dimer would split a neutral molecule
into charged fragments.

## Relaxing under pressure

`relax_cell_lengths` minimises *E* + *pV* over the three cell axes with molecules
held rigid at fixed fractional centroids and fixed orientation:

```python
from mmml.analysis.lattice_energy import relax_cell_lengths

relaxed = relax_cell_lengths(
    fixed.get_positions(), fixed.get_atomic_numbers(), fixed.cell.array,
    pressure_GPa=0.0, cutoff_A=12.0,
)
relaxed.cell_lengths_A, relaxed.e_lattice
```

The method is validated where the answer is already known — relaxing at the two
*measured* pressures:

| | CGenFF | Measured | Error |
|---|---|---|---|
| V at 1.33 GPa | 289.7 Å³ | 293.12 Å³ | −1.2% |
| V at 1.63 GPa | 286.1 Å³ | 285.46 Å³ | +0.2% |
| V at 0 GPa | 314.1 Å³ | 328.2 Å³ (153 K) | −4.3% |

Reproducing the measured compression to about a per cent means CGenFF has the
repulsive wall of this crystal roughly right, which is what the extrapolation to
zero pressure rests on. The zero-pressure cell comes out smaller than the 153 K
measurement, as a static calculation should.

This is not a full lattice relaxation: molecular orientation, internal geometry
and cell angles are all frozen, so the energy is an upper bound on the true
CGenFF minimum. Both approximations are reasonable starting from an experimental
structure whose molecules sit on crystallographic twofold axes; neither is exact.

## Sublimation enthalpy

With `ΔH_sub = −E_latt − 2RT` (see
[the acetone page](acetone-crystal-sublimation.md) for that convention):

| Cell | E_latt (kcal/mol) | ΔH_sub (kJ/mol) | vs experiment |
|---|---|---|---|
| as deposited, 1.33 GPa | −9.002 | 34.70 | −4.6% |
| relaxed to 0 GPa | −9.428 | 36.48 | **+0.3%** |

The experimental reference is a thermodynamic cycle, ΔH_vap + ΔH_fus =
30.2 + 6.16 = 36.4 kJ/mol near the melting point, both legs via the NIST
Chemistry WebBook, because no direct sublimation measurement for CH₂Cl₂ is
tabulated. Treat it as good to about a kJ/mol.

Do not over-read the final agreement: the `−2RT` convention assumes a rigid
molecule, the relaxation freezes orientation, and there is no zero-point term.
A few per cent is the resolution of this comparison. The *relative* result is the
robust one — relaxing to ambient pressure moves the answer from 4.6% off to 0.3%
off, so the deposited structures underbind because of the pressure they were
measured at, not because CGenFF is wrong about cohesion.

## Validating learned LJ scales

Neither this crystal nor its sublimation enthalpy appears in any hybrid ML/MM
training set, which makes it a genuine out-of-sample check on learned per-type LJ
scales:

```bash
DCM_SCALES=/path/to/hybrid_mm.json bash examples/dcm_crystal/run_all.sh
```

Both `crystal_lattice_energy` and `relax_cell_lengths` accept `sigma_scale` and
`epsilon_scale` directly. See [hybrid MM LJ scales](hybrid-mm-lj-scales.md).

## Limitations

- **No ambient-pressure experimental structure.** Everything at zero pressure is
  a relaxation, judged against a 1973 cell and nothing else.
- **Rigid-molecule relaxation.** Orientation, internal geometry and cell angles
  are frozen; the relaxed energy is an upper bound.
- **Static, classical.** No thermal expansion, no zero-point energy, no
  intramolecular contribution to ΔH_sub.
- **Rebuilt hydrogens.** Necessary, but it does import two constants from
  spectroscopy rather than from these diffraction experiments.
- **CGenFF has no σ-hole.** The Cl···Cl conclusion is therefore a statement about
  what dispersion alone can account for, corroborated here by the independent DFT
  study rather than proved.

## API

- `mmml.analysis.dcm_crystal` — `DCM_CRYSTAL_PHASES`, `read_dcm_phase`,
  `rebuild_methylene_hydrogens`, `halogen_contacts`, `h_cl_contacts`,
  `classify_halogen_motif`, `KAWAGUCHI_AMBIENT_CELL`,
  `DCM_SUBLIMATION_REFERENCE`
- `mmml.analysis.crystal_contacts` — `element_pair_contacts`,
  `normalize_hydrogen_positions`, `Contact`
- `mmml.analysis.lattice_energy` — `crystal_lattice_energy`,
  `decompose_lattice_energy_by_element_pair`, `relax_cell_lengths`,
  `sublimation_enthalpy_kcal_mol`

## See also

- [Solid acetone: crystal structure and sublimation enthalpy](acetone-crystal-sublimation.md)
  — the same machinery where five phases are available and all are at ambient
  pressure.
- [`build-crystal`](cli/commands/build-crystal.md) — writing these cells as PDB
  or supercells for use elsewhere.
