# Solid acetone: crystal structure and sublimation enthalpy

Builds crystalline acetone from published diffraction data, verifies it against
the distances the authors measured, and computes its sublimation enthalpy with
CGenFF. Everything runs on a CPU in under a minute — no CHARMM, no GPU, no
trained checkpoint, no MD.

```bash
bash examples/acetone_crystal/run_all.sh
```

Reference: D. R. Allan, S. J. Clark, R. M. Ibberson, S. Parsons, C. R. Pulham and
L. Sawyer, *The influence of pressure and temperature on the crystal structure of
acetone*, **Chem. Commun.** 1999, 751
([doi:10.1039/a900558g](https://doi.org/10.1039/a900558g)), CCDC 182/1197.
Coordinates via the Crystallography Open Database.

## The bundled structures

All five structures from the paper ship in `mmml/data/structures/`:

| `ACO_PHASE` | COD | Space group | a, b, c (Å) | Conditions | Z |
|---|---|---|---|---|---|
| `pbca_5k` | 7110465 | Pbca | 9.1669, 7.5323, 21.2486 | 5 K, neutron (d6) | 16 |
| `pbca_110k` | 7110466 | Pbca | 9.172, 7.761, 21.66 | 110 K, X-ray | 16 |
| `pbca_150k` | 7110464 | Pbca | 8.873, 8.000, 22.027 | 150 K, X-ray | 16 |
| `cmcm_160k` | 7110463 | Cmcm | 6.514, 5.4159, 10.756 | 160 K, metastable | 4 |
| `cmcm_15kbar` | 7110462 | Cmcm | 6.1219, 5.2029, 10.244 | 293 K, 15 kbar | 4 |

```python
from mmml.analysis.acetone_crystal import read_acetone_phase

atoms = read_acetone_phase("pbca_150k")   # 160 atoms, symmetry already applied
```

Two traps are handled for you. The 5 K structure was refined on acetone-d6 and
ASE preserves the deuterium masses, which inflates the density by 10%;
`read_acetone_phase` resets them unless you pass `protiate=False`. The 15 kbar
structure has rotationally disordered methyls — 12 half-occupancy hydrogens per
molecule — so it carries `usable_for_mm = False` and the force-field steps refuse
it rather than building a nonsense topology.

`mmml build-crystal --literature aco` maps a phase onto CHARMM `ACO` atom names
and writes a PDB with a correct `CRYST1`; see
[`build-crystal`](cli/commands/build-crystal.md).

## Verifying a built cell

Lattice parameters and molecule counts can all be right while the structure is
still wrong — a mis-applied symmetry operator, or molecules broken across a cell
face. The check that catches this is recomputing the intermolecular contacts the
authors measured:

```python
from mmml.analysis.acetone_crystal import carbonyl_contacts, ch_o_contacts

for contact in carbonyl_contacts(atoms, max_distance_A=3.8):
    print(contact.distance_A, contact.angle_deg, contact.motif)
```

All 15 distances quoted across the paper's five structures come back within
0.01 Å, and the motifs classify as described: Type II antiparallel and Type I
perpendicular in Pbca, Type III sheared-parallel in Cmcm.

The contacts also tighten monotonically on cooling — shortest H···O of 2.617,
2.511 and 2.336 Å at 150, 110 and 5 K — which is the structural change the paper
proposes as the origin of the broad heat-capacity anomaly near 127 K, unexplained
since Kelley's 1929 calorimetry. The 5 K row is neutron-derived and so carries
physically longer C–H bonds than the X-ray rows; part of that last step is
method rather than temperature.

## Lattice energy

```python
from mmml.analysis.lattice_energy import crystal_lattice_energy

result = crystal_lattice_energy(
    atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array,
    cutoff_A=12.0,
)
print(result.e_lattice, result.sublimation_enthalpy(150.0))   # kcal/mol
```

`mmml.analysis.lattice_energy` sums the CGenFF **intermolecular** energy over
explicit lattice translations, carrying the full 3×3 cell throughout. Three
choices are worth knowing about:

**Explicit lattice sums, not minimum image.** The MIC caps the cutoff at half the
shortest cell edge — 3.77 Å for the 5 K acetone cell — which is useless for both
dispersion and the real-space Ewald split. Summing over lattice shifts decouples
the cutoff from the cell size at negligible cost for a few hundred atoms.

**Intermolecular only.** Bonded and intramolecular nonbonded terms cancel exactly
against an isolated molecule frozen at the same geometry, so the lattice energy
is directly comparable to a gas-phase reference without ever computing one.

**Ewald, not truncation.** Coulomb between dipoles is conditionally convergent,
so the electrostatic term uses `build_kspace_integers` / `ewald_reciprocal_energy`
with tinfoil boundary conditions. It is independent of the cutoff to four
decimals, as it must be. Dispersion gets an analytic isotropic tail correction:
between 8 and 16 Å the bare LJ term moves 0.56 kcal/mol while LJ + tail moves
0.02.

The Ewald path is anchored against the analytic NaCl Madelung constant on both a
cubic and a deliberately non-cubic cell in `tests/unit/test_lattice_energy.py`.

## Sublimation enthalpy

```
dH_sub(T) = -E_latt - 2RT
```

The −2RT is the difference between the gas-phase 4RT (3/2 RT translation +
3/2 RT rotation + RT for pV) and the 6RT carried classically by the six
rigid-body lattice modes that replace them. It assumes a rigid molecule with the
same intramolecular vibration in both phases, and is classical throughout — no
zero-point term.

| Phase | T (K) | E_latt (kcal/mol) | ΔH_sub (kJ/mol) |
|---|---|---|---|
| `pbca_150k` | 150 | −10.99 | 43.5 |
| `pbca_110k` | 110 | −11.21 | 45.1 |
| `pbca_5k` | 5 | −11.24 | 46.9 |

There is no directly tabulated sublimation enthalpy for acetone, so the
comparison value is assembled from a cycle: ΔH_vap = 32.9 kJ/mol at 228 K
(Stephenson & Malanowski 1987) plus ΔH_fus = 5.72 kJ/mol at 176.6 K (Kelley
1929), giving ≈ 38.6 kJ/mol near the melting point. CGenFF overbinds by 13% at
150 K, rising to 21% at 5 K. That is expected for parameters fit to liquid
densities and heats of vaporisation — nothing in the fit ever saw a crystal — and
the reference sits near 180 K while these structures are colder.

The trend is the more interesting result: ΔH_sub rises 8% from 150 K to 5 K, so
the force field reproduces the paper's claim that the contacts strengthen on
cooling. Each row is evaluated at its own experimental geometry, so the trend is
inherited from the diffraction data rather than predicted.

## Testing learned LJ scales

Sublimation enthalpy is an observable hybrid ML/MM training never sees, which
makes it a real test of learned per-type LJ scales rather than a re-read of
training loss:

```bash
ACO_SCALES=artifacts/lj_scales/ckpts/.../hybrid_mm.json \
  uv run python examples/acetone_crystal/05_sublimation.py
```

`crystal_lattice_energy` also takes `sigma_scale` / `epsilon_scale` directly. See
[Trainable hybrid MM LJ scales](hybrid-mm-lj-scales.md) for producing the sidecar.

## Limitation: no crystal MD

These cells are orthorhombic but strongly non-cubic, and mmml's periodic MD paths
are cubic-only:

- `prepare_charmm_pbc` installs a cubic CHARMM IMAGE via `crystal.define_cubic`;
  there is no `define_ortho` call site in the package.
- `md-system` box resolution reduces a cell to one side through
  `cubic_box_side_from_cell`, which averages the edges. For the 5 K acetone cell
  that is a 12.65 Å cube.

That function now emits a `RuntimeWarning` when it averages a non-cubic cell, so
the failure mode is audible rather than silent, but the underlying restriction
stands. This workflow therefore stops at the static lattice energy, which is
correct and converged.

Supporting crystal MD would need, in rough order: an orthorhombic
`prepare_charmm_pbc`; a box-resolution path carrying `(a, b, c)` rather than `L`
through `resolve_handoff_box` and the ASE/JAX-MD suites; per-axis nonbonded
cutoffs; and a full cell threaded through the PME call sites, where
`box_length_from_cell` currently returns `a` alone. The MIC math in
`pbc_utils_jax` already handles a general cell, so the work is plumbing rather
than physics.

Finite-temperature ΔH_sub — `<U_gas> − <U_cryst>/N + RT` from NVT averages — and
a full free energy of sublimation via Einstein-crystal thermodynamic integration
both sit behind that same limitation.
