# Crystalline dichloromethane: cohesion and sublimation

Tests a published claim about what holds solid CH₂Cl₂ together, then relaxes the
crystal to ambient pressure and compares against experiment. No CHARMM, no GPU,
no trained checkpoint, no MD — the whole ladder runs on a CPU in under a minute.

```bash
bash examples/dcm_crystal/run_all.sh
```

## The claim under test

> M. Podsiadło, K. F. Dziubek and A. Katrusiak, *In situ high-pressure
> crystallization and compression of halogen contacts in dichloromethane*,
> **Acta Crystallogr. B** 61, 595 (2005).
> [doi:10.1107/S0108768105017374](https://doi.org/10.1107/S0108768105017374),
> CCDC [doi:10.5517/cc9lyjb](https://doi.org/10.5517/cc9lyjb).

The paper closes with a statement about energy rather than geometry:

> the crystal cohesion forces are dominated by H···Cl interactions rather than by
> Cl···Cl attractions

The authors reached it indirectly — from how the contacts compress and from the
crystal habit — because diffraction measures positions, not energies. A force
field measures energies directly, so step 03 simply checks it. CGenFF agrees:
H···Cl carries **63%** of the binding, Cl···Cl **16%**, and that 16% is entirely
dispersion (its electrostatic part is *repulsive*). A later plane-wave DFT study
that does model the σ-hole reached the same conclusion — Kurzydłowski, Chumak and
Rogoża, *Crystals* **10**, 920 (2020),
[doi:10.3390/cryst10100920](https://doi.org/10.3390/cryst10100920).

## The structures, and the one that is missing

These are the only two pure CH₂Cl₂ entries in the Crystallography Open Database,
both from the paper above, bundled under `mmml/data/structures/`:

| `DCM_PHASE` | COD | Space group | a, b, c (Å) | Conditions | Notes |
|---|---|---|---|---|---|
| `pbcn_133gpa` | 2100014 | Pbcn | 3.984, 7.863, 9.357 | 293 K, 1.33 GPa | **default** |
| `pbcn_163gpa` | 2100015 | Pbcn | 3.924, 7.793, 9.335 | 293 K, 1.63 GPa | used by the `dcm` `build-crystal` preset |

Both were grown in situ in a diamond-anvil cell, and **both are compressed**, by
10.7% and 13.0% respectively. That is the single most important fact about this
system: a crystal squeezed 11% below its ambient volume sits well up its
repulsive wall, so its static lattice energy is not a cohesive energy and must
not be compared with a sublimation enthalpy as it stands.

The ambient-pressure structure is isostructural (Pbcn, Z = 4) and was determined
by Kawaguchi, Tanaka, Takeuchi and Watanabé, *Bull. Chem. Soc. Jpn.* **46**, 62
(1973), [doi:10.1246/bcsj.46.62](https://doi.org/10.1246/bcsj.46.62). It predates
CIF deposition and has no openly licensed coordinates, so only its cell — 4.249,
8.138, 9.492 Å at ~153 K — is recorded here, as `KAWAGUCHI_AMBIENT_CELL`. Step 05
relaxes the crystal to zero pressure and checks the result against it.

## Two things that had to be fixed first

**The hydrogens.** X-rays scatter from electrons and a hydrogen has one bonding
electron, so the two refinements put C–H at 1.01(10) and 1.13(12) Å and disagree
on the hydrogen *direction* by a comparable amount. Both spreads are larger than
the compression between the two structures, and taken at face value they make the
shortest H···Cl contact appear to **lengthen** under pressure. Since CH₂Cl₂ is
C₂ᵥ and its carbon and chlorines are located to a few thousandths of an Ångström,
`rebuild_methylene_hydrogens` regenerates the hydrogens from the heavy-atom frame
plus two spectroscopic constants. This moves the lattice energy by ~0.2 kcal/mol
and restores the physical sign of the contact trend.

**The pressure.** `relax_cell_lengths` minimises *E* + *pV* over the three cell
axes with molecules held rigid. Relaxing at the two *measured* pressures
reproduces the measured volumes to −1.2% and +0.2%, which is the real test of the
method because those answers were already known. Relaxing to zero pressure then
gives a cell within 4.3% of the 1973 measurement (smaller, as a static
calculation should be against a 153 K one) and a lattice energy that can honestly
be turned into a sublimation enthalpy.

## The ladder

| Step | What it does |
|---|---|
| `00_check_env.py` | ASE, SciPy, JAX x64, both CIFs, CGenFF `RESI DCM` |
| `01_phases.py` | Checks both cells, volumes, densities and Z against the paper; states how compressed they are |
| `02_contacts.py` | Cl···Cl contacts with Desiraju–Parthasarathy typing, H···Cl contacts, and the hydrogen problem |
| `03_cohesion.py` | Splits the lattice energy over molecule pairs — the paper's claim, tested |
| `04_lattice_energy.py` | Lattice energy with a cutoff sweep showing the tail correction and Ewald are converged |
| `05_relax_and_sublimation.py` | Relaxes at 1.33, 1.63 and 0 GPa; compares cells and ΔH_sub against experiment |

Every step exits non-zero if its own check fails, so `run_all.sh` is a
regression test as much as a demonstration.

## Results

Starting from the 1.33 GPa structure, with rebuilt hydrogens, 12 Å cutoff:

| Quantity | CGenFF | Experiment | Error |
|---|---|---|---|
| V at 1.33 GPa | 289.7 Å³ | 293.12 Å³ | −1.2% |
| V at 1.63 GPa | 286.1 Å³ | 285.46 Å³ | +0.2% |
| V at 0 GPa | 314.1 Å³ | 328.2 Å³ (153 K) | −4.3% |
| ΔH_sub at 178 K | 36.5 kJ/mol | 36.4 kJ/mol | +0.3% |

The experimental ΔH_sub is itself a thermodynamic cycle, ΔH_vap + ΔH_fus =
30.2 + 6.16 kJ/mol, both via the NIST Chemistry WebBook, because no direct
sublimation measurement for CH₂Cl₂ is tabulated. Treat it as good to about a
kJ/mol.

Do not over-read the final agreement. The `−2RT` convention assumes a rigid
molecule, the relaxation freezes molecular orientation and cell angles, and there
is no zero-point term anywhere. A few per cent is the resolution of this
comparison; both numbers sit inside it. What the ladder does establish is the
*relative* claim, which is much more robust: relaxing to ambient pressure moves
the answer from 4.6% off to 0.3% off, so the deposited structures underbind
because of the pressure they were measured at, not because CGenFF is wrong about
cohesion.

## Knobs

Set before `run_all.sh`, or see `_env.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `DCM_PHASE` | `pbcn_133gpa` | Which deposited structure to work from |
| `DCM_CUTOFF` | `12.0` | Real-space cutoff (Å) for the LJ sum and the Ewald split |
| `DCM_TEMPERATURE` | `178.2` | Temperature for the `−2RT` term, at the melting point |
| `DCM_SCALES` | *(unset)* | `hybrid_mm.json` with learned per-type LJ scales |
| `ARTIFACTS_DIR` | `artifacts/dcm_crystal` | Where the relaxed ambient structure is written |

`DCM_SCALES` makes this an out-of-sample check on hybrid ML/MM training: neither
the sublimation enthalpy nor the crystal is in any training set. See
[`examples/lj_scales`](../lj_scales) and `docs/hybrid-mm-lj-scales.md`.

## See also

- [`examples/acetone_crystal`](../acetone_crystal) — the same ladder for acetone,
  where five phases are available and all of them are at ambient pressure.
- `mmml build-crystal --literature dcm` — writes the 1.63 GPa cell as a PDB or
  supercell for use elsewhere.
