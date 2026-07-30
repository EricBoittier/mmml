# Liquid DCM / ACO density sweep

Validation scripts for liquid **dichloromethane (DCM)** and **acetone (ACO)** at
**0.50, 0.75 and 1.00 × bulk density**, exercised across every MD backend,
long-range electrostatics solver, and ML/MM partitioning the package supports.

The density axis is the point: sub-bulk cells are easy and forgiving, bulk cells
are where truncated electrostatics, switching widths and ML/MM double-counting
actually bite. Running the same settings matrix at all three densities separates
"this setting is wrong" from "this cell is just dilute".

## Layout

| Script | What it does |
|--------|--------------|
| `scripts/01_build_boxes.sh` | Packmol + MC + CHARMM SD/ABNR box build (`mmml liquid-box`, MM only, no GPU/checkpoint) |
| `scripts/02_backends.sh` | Same cell through `ase`, `jaxmd`, `pycharmm` |
| `scripts/03_electrostatics.sh` | `mic`, `ewald` (±`--ewald-omit-self`), `jax_pme` × {ewald,pme,p3m}, `nvalchemiops_pme`, `scafacos` |
| `scripts/04_ml_mm.sh` | ML/MM partitioning, hybrid charge modes, switching, MM pair provider |
| `scripts/run_all.sh` | All of the above, with an `mmml doctor` preflight |
| `scripts/print_density_table.py` | Regenerate the monomer-count table below |

All scripts honour `DRY_RUN=1` (print commands instead of running), plus
`BOX_SIZE`, `SOLVENTS`, `FRACTIONS`, `TEMPERATURE`, `DT_FS`, `PS_PROD`,
`OUT_ROOT`.

```bash
export MMML_CKPT=/path/to/DESdimers_params.json
cd workflows/liquid_density_sweep/scripts

DRY_RUN=1 ./run_all.sh        # inspect the matrix first
./01_build_boxes.sh           # MM-only: cheap, no checkpoint needed
./run_all.sh                  # full sweep
```

## The matrix

Monomer counts are computed with mmml's own sizing helper
(`box_sizing.n_molecules_for_target_density_in_fixed_box`), so they match what
`md-system` / `liquid-box` build. Bulk densities at ~298 K: **DCM 1.326 g/cm³**
(5 atoms/molecule), **ACO 0.784 g/cm³** (10 atoms/molecule).

| L (Å) | Solvent | 0.50 × ρ | 0.75 × ρ | 1.00 × ρ |
|-------|---------|----------|----------|----------|
| 28 | DCM | 103 (515 at) | 155 (775 at) | 206 (1030 at) |
| 28 | ACO | 89 (890 at) | 134 (1340 at) | 178 (1780 at) |
| 32 | DCM | 154 (770 at) | 231 (1155 at) | 308 (1540 at) |
| 32 | ACO | 133 (1330 at) | 200 (2000 at) | 266 (2660 at) |
| 36 | DCM | 219 (1095 at) | 329 (1645 at) | 439 (2195 at) |
| 36 | ACO | 190 (1900 at) | 284 (2840 at) | 379 (3790 at) |

Counts can also be derived at run time instead of hard-coded:

```bash
mmml md-system --composition DCM:1 --box-auto count --box-size 28 \
  --bulk-density-fraction 0.75 ...
```

## ⚠️ Before submitting: MLpot atom limits

`mmml doctor` prints, near the end:

```
CHARMM MLpot limits: max_Nml=..., max_Npr=...
  source: ...
```

If `source` says **`conservative fallback (libcharmm.so older than api_func.F90)`**
then the limits collapse to `max_Nml=100`, `max_Npr=100000`. **Every cell in
this matrix exceeds that** — the smallest, DCM:103 at L=28, is already 515 ML
atoms — and jobs abort at setup with

> `CHARMM MLpot supports at most 100 ML atoms in this libcharmm build`

`setup/charmm/source/api/api_func.F90` actually declares `max_Nml = 50000` and
`max_Npr = 128000000`; the fallback triggers purely because `libcharmm.so` is
older than the header. Rebuild once per node:

```bash
./scripts/rebuild_charmm_mlpot.sh --clean
mmml doctor    # confirm `source: api_func.F90`, not the fallback
```

Two more things `mmml doctor` will tell you:

- **libcharmm is MPI-linked** → launch MLpot runs through
  `scripts/mmml-charmm-mpirun.sh` (the scripts here already do).
- **cupy ≥ 14 + gpu4pyscf** → DFT Hessians abort with a `c_contiguous`
  assertion; pin `cupy>=13,<14`. Irrelevant unless a job does QM.

## Choosing settings

**Electrostatics.** `mic` truncates; `ewald` is full-box and pure JAX;
`jax_pme` adds k-space with selectable method. `nvalchemiops_pme` and
`scafacos` need `--mm-nonbond-mode periodic_external` (external Coulomb +
CHARMM IMAGE VDW), so they imply a `pbc_*` setup on the `pycharmm` backend.

Match the operator to the model: `--lr-solver ewald` for Ewald-trained
checkpoints, `--lr-solver ewald --ewald-omit-self` for MIC-trained ones. Mixing
them biases energies silently.

**Hybrid MM charges.** `latent` / `q1` is **dimer-only** and invalid for an
N-monomer liquid. Use `fixed`, `q0`, `latent_mean`, or `latent_dynamic`, which
are defined for any `n_monomers`. See [`docs/hybrid-mm-charges.md`](../../docs/hybrid-mm-charges.md).

**Mixed boxes.** DCM + ACO together needs an interaction policy so no pair is
double-counted — see [`docs/md-interaction-policies.md`](../../docs/md-interaction-policies.md)
and the `examples/interaction_policy_*.yaml` starting points.

## Interpreting the sweep

- Run **`--setup pbc_nve`** per variant as the correctness probe: a
  solver/cutoff/switching mismatch shows up as energy drift.
- Run **`--setup pbc_npt`** as the physics probe: started at 0.50/0.75/1.00 ×
  ρ_bulk, does each cell relax back toward ρ_bulk? That is the real validation
  that the hybrid model reproduces the liquid, rather than being held there.
- Compare energies of the **same starting configuration** across solvers before
  comparing trajectories — that isolates the operator from dynamics noise.
- `mic` vs `ewald` should diverge more at 1.00 × ρ than at 0.50 × ρ. If they
  agree everywhere, the cell is too dilute to be testing long range.

## Related workflows

- [`pbc_liquid_density_dyn`](../pbc_liquid_density_dyn/) — Snakemake matrix,
  persistent PyCHARMM production MD with resilient density prep ladders.
- [`unified_backend_sweep`](../unified_backend_sweep/) — backend/setting sweep
  with NVE + force validation and result collection.
- [`dcm_density_setup_compare`](../dcm_density_setup_compare/) — DCM setup-path
  comparison.

Outputs land in `artifacts/` and are gitignored.
