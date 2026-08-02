# NpT density, ΔH_vap and ΔG — validating the hybrid potential outside the loss

Branch: `npt-density-dh-dg`.

Everything the DES LJ-scale work has produced so far is measured *inside* the
training objective — energies and forces on dimers. That cannot close
`solvent_burst_default_matrix: status: unverified` in the evidence registry,
because a model fitted to those numbers will reproduce them. This branch adds
the three condensed-phase observables that sit outside the loss.

| Observable | Tests | Status |
|---|---|---|
| **Density** | the repulsive/attractive balance through the equation of state | tooling done, not run |
| **ΔH_vap** | the intermolecular energy directly | tooling done, not run |
| **ΔG** | the full free energy, entropy included | designed, not built |

---

## Why NpT and not the box we already have

A Packmol box is constructed **at** a target density — 732 waters in a 28.0 Å
cube is 0.9975 g/cm³ because that is what it was asked for. Its density is an
*input*. Only an NpT run at fixed pressure lets the box find its own volume, at
which point density becomes a measurement of the potential.

## Pressure units

`npt_nose_hoover` takes pressure in the same units as energy/volume — eV/Å³ in
the metal system, not bar or atm. The conversion in `jaxmd_runner.py` is

```
pressure = p_atm * 1.01325 * unit['pressure']
```

which gives **6.324209e-07 eV/Å³** for 1 atm against an exact
101325 / (1.602176634e-19 / 1e-30) = 6.324209069697369e-07. Verified to zero
relative error. Getting this wrong by the 1.01325 bar/atm factor alone would
bias the equilibrium volume by ~1.3%, which is the size of the effect being
measured.

## What is built

**`scripts/analyze_npt_density.py`** — equilibrated density from the
`density_g_cm3` series the NpT runner already records. It refuses three things
rather than reporting them quietly:

- **an unequilibrated average** — the production window's least-squares slope is
  compared against its own standard error, and a drift significant at 2σ is
  reported as `NOT EQUILIBRATED` instead of being averaged away;
- **the naive standard error** — NpT density is strongly autocorrelated and the
  naive SEM understates the bar by roughly `sqrt(2 tau / dt)`; errors come from
  block averaging;
- **a mismatched reference** — density is strongly temperature dependent, so a
  reference quoted more than 5 K from the run is flagged.

It also errors out on a trajectory with no `density_g_cm3` (an NVT/NVE run
cannot measure density — its volume is fixed by construction) and on any
non-finite value (a diverged barostat must not be averaged).

**`scripts/slurm/run_npt_density_campaign.sh`** — `--setup pbc_npt --backend
jaxmd` at 1 atm on the certified boxes, then the analysis. Note there is no
`--ensemble` flag: the ensemble is the `--setup` preset, and `--n-equil` is
`lambda_ti`-only, so equilibration is handled by running `EQ_PS + PROD_PS` and
discarding the leading fraction in analysis — which is also the only place drift
is actually tested.

Reference densities: water **0.99705** (298.15 K), methanol **0.78660**
(298.15 K), ammonia **0.68190** (239.82 K). Ammonia runs at 240 K because it
boils at 239.8 K — a 298 K "liquid ammonia" box is a gas.

## ΔG — designed, not built

Density and ΔH_vap are both *energetic*; neither constrains entropy. ΔG does,
and it is the quantity a force field is ultimately judged on.

The tractable target is **hydration free energy** of a solute in the TIP3 box,
by alchemical decoupling. The repository already has the pieces:

- `--setup lambda_ti` — alchemical TI with per-window minimisation
- `mmml lambda-mbar` — MBAR over the windows
- `mmml umbrella-sample` / `umbrella-mbar` — the restrained-sampling path

What is missing is a defined λ schedule for *decoupling* rather than the
existing reaction-coordinate use, and the soft-core treatment that keeps the
endpoints finite. Two known traps before anyone starts:

1. **`lr_solver: mic` is mandatory for trained LJ scales**, so the free energy
   inherits truncated-MIC electrostatics. A hydration free energy is
   long-range-dominated and will carry a finite-size error that a Ewald run
   would not. That must be quantified before any ΔG is quoted.
2. **Decoupling LJ without soft-core** gives a divergent dU/dλ at the vanishing
   endpoint. jax-md does not provide soft-core LJ out of the box.

ΔG should not be attempted until density and ΔH_vap agree with experiment — if
the potential misses the density, its free energies are not interesting.

## Status

Nothing here has been run. The TIP3 box exists; MEOH and AMM1 are still
building. No NpT trajectory has been produced, so no density, ΔH_vap or ΔG
number exists yet.
