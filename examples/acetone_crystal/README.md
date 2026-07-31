# Solid acetone in periodic boundary conditions

Builds crystalline acetone from its published structure and computes the
sublimation enthalpy with CGenFF. No CHARMM, no GPU, no trained checkpoint, no
MD — the whole ladder runs on a CPU in under a minute.

```bash
bash examples/acetone_crystal/run_all.sh
```

## The structures

All five come from one paper:

> D. R. Allan, S. J. Clark, R. M. Ibberson, S. Parsons, C. R. Pulham and
> L. Sawyer, *The influence of pressure and temperature on the crystal structure
> of acetone*, **Chem. Commun.** 1999, 751.
> [doi:10.1039/a900558g](https://doi.org/10.1039/a900558g), CCDC 182/1197.

Deposited coordinates are redistributed by the Crystallography Open Database and
are bundled under `mmml/data/structures/`:

| `ACO_PHASE` | COD | Space group | a, b, c (Å) | Conditions | Notes |
|---|---|---|---|---|---|
| `pbca_5k` | 7110465 | Pbca | 9.1669, 7.5323, 21.2486 | 5 K | neutron powder, acetone-d6 |
| `pbca_110k` | 7110466 | Pbca | 9.172, 7.761, 21.66 | 110 K | X-ray |
| `pbca_150k` | 7110464 | Pbca | 8.873, 8.000, 22.027 | 150 K | X-ray, **default** |
| `cmcm_160k` | 7110463 | Cmcm | 6.514, 5.4159, 10.756 | 160 K | metastable |
| `cmcm_15kbar` | 7110462 | Cmcm | 6.1219, 5.2029, 10.244 | 293 K, 15 kbar | methyls disordered |

The Pbca entries carry Z = 16, so ASE's symmetry expansion turns a two-molecule
asymmetric unit into 160 atoms. `pbca_150k` is the default because it is the
stable low-temperature phase with ordered, refined hydrogens.

The 15 kbar phase has rotationally disordered methyls — 12 half-occupancy
hydrogens per molecule — so it can be used for packing analysis but not for a
force field until one rotamer is chosen. Step 03 refuses it with that message
rather than producing a nonsense topology.

## The ladder

| Step | What it does |
|---|---|
| `00_check_env.py` | ASE, SciPy, JAX x64, the five CIFs, CGenFF `RESI ACO` |
| `01_phases.py` | Reads each cell; checks lattice parameters, volume and Z against the paper |
| `02_contacts.py` | Recomputes the published C=O···C=O and C–H···O distances |
| `03_build_supercell.sh` | Writes PDB (with `CRYST1`) and extxyz via `mmml build-crystal` |
| `04_lattice_energy.py` | Lattice energy with a cutoff convergence study |
| `05_sublimation.py` | ΔH_sub for the three Pbca temperatures, against experiment |

Step 03 is skipped by `run_all.sh` unless `ACO_BUILD=1`, since nothing
downstream consumes its output.

## Why the structural check is the important one

A cell can have the right lattice parameters and the right molecule count and
still be wrong — a mis-applied symmetry operator, or molecules broken across a
face. Step 02 catches that by recomputing what the authors actually measured.
All 15 distances quoted in the paper come back within 0.01 Å, for example at
5 K:

```
type_ii_antiparallel       3.231 Å   found 3.231 Å
type_i_perpendicular       3.391 Å   found 3.392 Å
ch_o_between_chains        2.336 Å   found 2.336 Å
```

The contacts also tighten monotonically on cooling (H···O of 2.617, 2.511,
2.336 Å at 150, 110 and 5 K), which is the structural change the paper proposes
as the origin of the broad heat-capacity anomaly near 127 K that had been
unexplained since Kelley's 1929 calorimetry.

## How the energy is computed

`mmml.analysis.lattice_energy` sums the CGenFF intermolecular energy over
explicit lattice translations, carrying the full 3×3 cell throughout:

- **Dispersion/repulsion**: CHARMM Lennard-Jones with an analytic isotropic tail
  correction beyond the cutoff. Between 8 and 16 Å the bare LJ term moves by
  0.56 kcal/mol while LJ + tail moves by 0.02.
- **Electrostatics**: Ewald summation with tinfoil boundary conditions, using
  `build_kspace_integers` / `ewald_reciprocal_energy`, which take a general cell.
  Truncated Coulomb would be conditionally convergent between dipoles. The
  electrostatic term is independent of the cutoff to four decimals, as it must be.
- **Only intermolecular pairs** are summed, so bonded and intramolecular
  nonbonded terms cancel exactly against a gas molecule at the same geometry.
  That is why no separate gas calculation appears anywhere.

Sublimation enthalpy uses the standard rigid-molecule result

```
dH_sub(T) = -E_latt - 2RT
```

where −2RT is the gap between the gas-phase 4RT (3/2 RT translation + 3/2 RT
rotation + RT for pV) and the 6RT of the six rigid-body lattice modes that
replace them. It is classical and neglects the difference in intramolecular
vibration between phases.

## Results with stock CGenFF

| Phase | T (K) | E_latt (kcal/mol) | ΔH_sub (kJ/mol) |
|---|---|---|---|
| `pbca_150k` | 150 | −10.99 | 43.5 |
| `pbca_110k` | 110 | −11.21 | 45.1 |
| `pbca_5k` | 5 | −11.24 | 46.9 |

Experiment has no directly tabulated sublimation enthalpy for acetone, so
step 05 assembles one from a cycle: ΔH_vap = 32.9 kJ/mol at 228 K (Stephenson &
Malanowski 1987) plus ΔH_fus = 5.72 kJ/mol at 176.6 K (Kelley 1929), giving
≈ 38.6 kJ/mol near the melting point.

CGenFF therefore overbinds the crystal by roughly 15%, which is unsurprising:
its parameters were fit to liquid densities and heats of vaporisation, and
nothing in that fit ever saw a crystal. Part of the gap is also temperature —
the reference sits near 180 K while these structures are colder.

More interesting than the offset is the trend. ΔH_sub rises 8% from 150 K to
5 K, so the force field reproduces the paper's central claim that the contacts
strengthen on cooling. Each row is evaluated at its own experimental geometry,
so that trend is inherited from the diffraction data rather than predicted.

## Testing learned LJ scales against a crystal

Sublimation enthalpy is an observable that hybrid ML/MM training never sees, so
it is a genuine test of learned per-type LJ scales rather than a re-read of
training loss:

```bash
ACO_SCALES=artifacts/lj_scales/ckpts/.../hybrid_mm.json \
  uv run python examples/acetone_crystal/05_sublimation.py
```

Compare against a stock run to see whether training moved the crystal in the
right direction. See `examples/lj_scales/` for producing the sidecar.

## Limitation: no crystal MD yet

These cells are orthorhombic but strongly non-cubic (9.17 × 7.53 × 21.25 Å),
and mmml's periodic MD paths are cubic-only:

- `prepare_charmm_pbc` installs a cubic CHARMM IMAGE via `crystal.define_cubic`;
  there is no `define_ortho` call site.
- `md-system` box resolution reduces a cell to one side length through
  `cubic_box_side_from_cell`, which averages unequal edges. For the 5 K cell
  that is a 12.65 Å cube.

So this ladder stops at the static lattice energy, which is correct and
converged, rather than handing the structure to an MD path that would quietly
change its shape. `cubic_box_side_from_cell` now warns when it averages a
non-cubic cell, so the failure mode is at least audible.

Running crystal MD would need, in rough order: an orthorhombic
`prepare_charmm_pbc`, a box-resolution path carrying `(a, b, c)` instead of `L`,
per-axis nonbonded cutoffs, and a full cell threaded through the PME call sites
(`box_length_from_cell` currently returns `a` alone). The MIC math in
`pbc_utils_jax` already handles a general cell, so the work is in the plumbing
rather than the physics.

Finite-temperature ΔH_sub — `<U_gas> − <U_cryst>/N + RT` from NVT averages —
and a full free energy of sublimation via Einstein-crystal thermodynamic
integration both sit behind that same limitation.
