# NH₃–CH₃Cl PhysNet example (`examples/m`)

Checkpoint and filtered dataset from commit
[`30eb7a01f7fcf1d42a795f188526a80e547110fd`](https://github.com/EricBoittier/mmml/commit/30eb7a01f7fcf1d42a795f188526a80e547110fd):

| File | Role |
|------|------|
| `kl.json` | Portable PhysNet params (`natoms=9`, charges + ZBL, vacuum) |
| `nh3_ch3cl_filtered.npz` | 16 000 frames (`N=9` dimers + NH₃ / CH₃Cl monomers) |
| `top_ch3cl.rtf` | Append topology for CGenFF residue `CH3CL` (used by Packmol `md-system`) |
| `par_ch3cl.prm` | Append bonded params for `CG331`–`CLGA1` (missing from stock CGenFF) |

Docs report (after running the pipeline):
[`docs/examples/nh3-ch3cl-results.md`](../../docs/examples/nh3-ch3cl-results.md).

## Environment

```bash
cd /path/to/mmml
source examples/m/_env.sh
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `MMML_CKPT` | `examples/m/kl.json` | Checkpoint |
| `MMML_DATA` | `examples/m/nh3_ch3cl_filtered.npz` | Eval NPZ |
| `MMML_CGENFF_EXTRA_RTF` | `examples/m/top_ch3cl.rtf` | Enables `CH3CL` in compositions |
| `MMML_CGENFF_EXTRA_PRM` | `examples/m/par_ch3cl.prm` | Bonded params for append `CH3CL` |
| `ARTIFACTS_DIR` | `artifacts/nh3_ch3cl` | Outputs |

## Quick run (full report)

```bash
bash examples/m/run_all.sh
```

Steps:

1. `01_evaluate.sh` — `mmml physnet-evaluate --plots`
2. `run_md_smokes.sh` — free-space NVE/NVT
3. `02_figures_and_report.py` — house-style figures + MkDocs page

### Evaluate only

```bash
NUM_SAMPLES=256 bash examples/m/01_evaluate.sh
uv run python examples/m/02_figures_and_report.py
```

`01_evaluate.sh` builds a dimer-only (`N=9`) NPZ and runs
`physnet-evaluate --subtract-mean` (absolute QM energies in the NPZ are not on
the checkpoint’s energy scale; force/dipole errors are absolute).

### MD smokes

**ML-only Python** (no CHARMM; geometry from a dataset dimer frame):

```bash
uv run python examples/m/03_free_nve_ase.py --n-steps 40
uv run python examples/m/04_free_nvt_ase.py --n-steps 40
uv run python examples/m/05_free_nve_jaxmd.py --n-steps 40
uv run python examples/m/06_free_nvt_jaxmd.py --n-steps 40
```

Each run writes under `artifacts/nh3_ch3cl/free_*_{ase,jaxmd}/`:

| File | Format |
|------|--------|
| `md.traj` | ASE trajectory (energy, forces, velocities per frame) |
| `md.xyz` | multi-frame XYZ |
| `final.xyz` / `final.npz` | last frame |
| `md_summary.json` | energies / temperatures + artifact paths |

Use `--traj-interval N` to thin frames (default every step).

**`md-system` Packmol** (`AMM1:1,CH3CL:1`, `--include-mm` off; needs PyCHARMM for PSF):

```bash
source examples/m/_env.sh
uv run mmml md-system --config examples/m/yaml/free_nve_ase.yaml
uv run mmml md-system --config examples/m/yaml/free_nve_jaxmd.yaml
uv run mmml md-system --config examples/m/yaml/free_nve_pycharmm.yaml
uv run mmml md-system --config examples/m/yaml/free_nvt_ase.yaml
uv run mmml md-system --config examples/m/yaml/free_nvt_jaxmd.yaml
uv run mmml md-system --config examples/m/yaml/free_nvt_pycharmm.yaml
```

Skip CHARMM-backed legs: `RUN_MD_SYSTEM=0 bash examples/m/run_md_smokes.sh`  
or `RUN_PYCHARMM=0` to keep ASE/JAX-MD `md-system` only when PyCHARMM is present.

### Solvated boxes (`make-box`) + mechanical embedding

Export a CGenFF-named solute PDB from the NPZ, then solvate with
`mmml make-box` in **ACN**, **TIP3**, and **DMSO** (default **30 Å** cube):

```bash
source examples/m/_env.sh
uv run python examples/m/07_export_solute_pdb.py
# Smoke: fixed --n (12 molecules). Production density: USE_DENSITY=1
BOX_SIZE=30 bash examples/m/08_make_boxes.sh
# or: BOX_SIZE=30 USE_DENSITY=1 bash examples/m/08_make_boxes.sh
```

Outputs: `artifacts/nh3_ch3cl/boxes/{acn,tip3,dmso}/model.{pdb,psf}` + `box.json`.

**Load into `md-system` (30 Å TIP3, JAX-MD unified mechanical embedding):**

```bash
# One-shot: export → make-box → short PBC NVE
bash examples/m/run_sol_tip3_30A.sh

# Or stepwise after 08_make_boxes.sh:
uv run mmml md-system --config examples/m/yaml/sol_tip3_30A_md.yaml
uv run mmml md-system --config examples/m/yaml/mech_embed_from_box_tip3.yaml --run-all
```

**Mechanical embedding** = former `cg_jax` mode: ML monomers (`ml_intra`) + MM
intermolecular (`mm_nonbonded`), not ML–MM electrostatic embedding.

| Backend | Config |
|---------|--------|
| `jaxmd` + `jaxmd_unified: true` | Shared `mmml.md` (`ml_intra` + `mm_nonbonded`) |
| `ase` / `pycharmm` | Hybrid calculator with `include_mm: true` |

Composition campaigns (Packmol inside `md-system`; no make-box required).
Default **30 Å** PBC cube; solvent counts are smoke-sparse (`TIP3:12`, etc.) —
use the make-box path with `USE_DENSITY=1` for production-like filling.

```bash
uv run mmml md-system --config examples/m/yaml/mech_embed_tip3.yaml --run-all
uv run mmml md-system --config examples/m/yaml/mech_embed_acn.yaml --run-all
uv run mmml md-system --config examples/m/yaml/mech_embed_dmso.yaml --run-all
```

From make-box PDBs (after `08_make_boxes.sh`):

```bash
uv run mmml md-system --config examples/m/yaml/mech_embed_from_box_tip3.yaml --run-all
uv run mmml md-system --config examples/m/yaml/mech_embed_from_box_acn.yaml --run-all
uv run mmml md-system --config examples/m/yaml/mech_embed_from_box_dmso.yaml --run-all
```

One-shot: `bash examples/m/run_mech_embed_smokes.sh`  
(`RUN_MAKE_BOX=0` / `RUN_FROM_BOX=0` to skip those legs).

### Electrostatic embedding

MM Coulomb uses **ML charges** (cg_jax `peptide_water_electrostatic_embedding`
analogue). `kl.json` has `charges: true`.

| Config | Charge mode | System |
|--------|-------------|--------|
| `yaml/es_embed_{tip3,acn,dmso}.yaml` | `q0` (Q⁰, liquid-safe) | solute + solvent |
| `yaml/es_embed_dimer_latent.yaml` | `latent` / Q¹ | **dimer only** AMM1+CH3CL |
| `yaml/es_embed_tip3_latent_dynamic.yaml` | `latent_dynamic` | liquid heuristic |
| `yaml/es_embed_tip3_ewald.yaml` | `q0` + `lr_solver: ewald` | TIP3 |
| `yaml/es_embed_from_box_tip3.yaml` | `q0` from make-box PDB | after `08_make_boxes.sh` |

```bash
uv run mmml md-system --config examples/m/yaml/es_embed_tip3.yaml --run-all
uv run mmml md-system --config examples/m/yaml/es_embed_dimer_latent.yaml --run-all
bash examples/m/run_es_embed_smokes.sh
```

`q0` / `latent*` are refused with `jax_pme` — use `mic` or `ewald` (see
[`docs/calculator-capabilities.md`](../../docs/calculator-capabilities.md)).

### Ewald / long-range Coulomb (all options)

Full matrix on TIP3 (`yaml/ewald_all_tip3.yaml`); ACN/DMSO subsets in
`ewald_all_{acn,dmso}.yaml`. Fixed charges (`mm_charge_mode: fixed`).

| Job id pattern | `lr_solver` / mode | Backends |
|----------------|--------------------|----------|
| `mic_*` | `mic` (truncated MIC) | ase, jaxmd, pycharmm |
| `ewald_*` | `ewald` full-box hybrid | ase, jaxmd, pycharmm |
| `ewald_omit_self_*` | `ewald` + `--ewald-omit-self` | ase, jaxmd, pycharmm |
| `jax_pme_ewald_*` | `jax_pme` method=`ewald` | ase, jaxmd, pycharmm |
| `jax_pme_pme_pycharmm` | `jax_pme` method=`pme` | pycharmm |
| `jax_pme_p3m_pycharmm` | `jax_pme` method=`p3m` | pycharmm |
| `pe_ewald_pycharmm` | `periodic_external` + `ewald` | pycharmm |
| `pe_ewald_coulomb_only_pycharmm` | same, `periodic_charmm_vdw: false` | pycharmm |
| `pe_nvalchemiops_pycharmm` | `periodic_external` + `nvalchemiops_pme` | pycharmm (opt) |
| `pe_scafacos_pycharmm` | `periodic_external` + `scafacos` | pycharmm (opt) |

```bash
# Full TIP3 matrix (skips missing jax-pme / nvalchemiops / ScaFaCoS):
bash examples/m/run_ewald_smokes.sh

# One job:
uv run mmml md-system --config examples/m/yaml/ewald_all_tip3.yaml --job-id ewald_pycharmm
```

Set `SCAFACOS_LIB=/path/to/libfcs.so` for the ScaFaCoS leg. Optional:
`RUN_ACN=0 RUN_DMSO=0` to run only the TIP3 matrix.

## Reaction-path toolkit (gas + solution)

End-to-end SN₂-like workflows for NH₃–CH₃Cl. Regenerate gas-phase endpoints
from the bundled NPZ:

```bash
uv run python examples/m/07_export_neb_endpoints.py
# → examples/m/neb/reag_0_opt.xyz, prod_0_opt.xyz
```

| Method | Gas phase | Explicit solvent (TIP3 / ACN / DMSO) |
|--------|-----------|----------------------------------------|
| **`mmml umbrella-sample`** | `engine: packed_ml` — batched all-ML NVT | `engine: hybrid_jaxmd` — ML reactive complex + MM solvent ([`yaml/umbrella_nc_tip3.yaml`](yaml/umbrella_nc_tip3.yaml), `14_umbrella_sample_sol.sh`) |
| **ADUMB** (PyCHARMM adaptive umbrella) | `yaml/adumb_nc_distance.yaml`, `09_adumb_nc_distance.sh` | `yaml/adumb_nc_distance_{tip3,acn,dmso}.yaml`; `SOLVATED=1 SOLVENT=tip3 bash examples/m/09_adumb_nc_distance.sh` |
| **NEB** (ASE nudged elastic band) | `yaml/neb.yaml`, `13_neb.sh` | Gas-phase path only (same endpoints) |
| **DMC** (Diffusion Monte Carlo) | `15_dmc_basins.sh` on react/product XYZ | Gas-phase basins only (same endpoints) |

One-shot ML smokes (no CHARMM):

```bash
bash examples/m/run_reaction_path_smokes.sh
# RUN_ADUMB=1 to include PyCHARMM ADUMB vacuum + TIP3 legs
```

**Studix GPU campaign** (seeds × temperatures × solvents, checkpoint
`model_ext.json`): [`workflows/nh3_ch3cl_reaction_path/`](../../workflows/nh3_ch3cl_reaction_path/).

### Fixed-bias umbrella (`mmml umbrella-sample`)

**Gas (`engine: packed_ml`)** — batched distance umbrella with PhysNet + JAX-MD
Langevin NVT. Atom order: `Cl, N, C, H×3(N), H×3(C)` (same as NEB endpoints).

```bash
source examples/m/_env.sh
bash examples/m/14_umbrella_sample_gas.sh
# or:
uv run mmml umbrella-sample --config examples/m/yaml/umbrella_nc_gas.yaml --overwrite
# 2D Cl–C × N–C grid:
uv run mmml umbrella-sample --config examples/m/yaml/umbrella_clc_cn_2d_gas.yaml --overwrite
# MBAR post-processing:
uv run mmml umbrella-mbar --run-dir artifacts/nh3_ch3cl/umbrella_nc_gas
```

**Solution (`engine: hybrid_jaxmd`)** — mechanical embedding: ML on the reactive
AMM1+CH3CL complex only; TIP3 (or other) solvent as MM. Per-window
`JaxmdDriver` NVT (not packed). Needs a make-box PSF/PDB (30 Å TIP3 by default).

```bash
source examples/m/_env.sh
bash examples/m/14_umbrella_sample_sol.sh
# or after 08_make_boxes.sh:
uv run mmml umbrella-sample --config examples/m/yaml/umbrella_nc_tip3.yaml --overwrite
uv run mmml umbrella-mbar --run-dir artifacts/nh3_ch3cl/umbrella_nc_tip3
```

Writes `umbrella_snapshots.npz` (includes `energies_unbiased_ev` +
`ml_atom_indices` for hybrid), `umbrella_summary.json`, and
`umbrella_bin_minima.traj` under `artifacts/nh3_ch3cl/umbrella_*`.

### NEB (ASE nudged elastic band)

Vacuum SN2-like path for NH₃–CH₃Cl with `kl.json` (endpoints under `neb/`):

```bash
source examples/m/_env.sh
# Smoke (11 images):
bash examples/m/13_neb.sh
# Or via YAML:
uv run mmml neb --config examples/m/yaml/neb.yaml --overwrite
# Dense band (~Asparagus 99-image setup):
N_IMAGES=99 bash examples/m/13_neb.sh
```

Writes `artifacts/nh3_ch3cl/neb/{neb.traj,neb.xyz,neb_profile.dat,neb_plot.png,neb_summary.json}`.
Profile columns: reaction coordinate (Å), ΔE (kcal/mol), N–C and Cl–C distances.

Docs: [`docs/neb.md`](../../docs/neb.md).

### DMC (reactant / product basins)

Vibrational ground-state estimates at the exported basin geometries (gas phase,
9 atoms). Requires N and Cl support in `mmml dmc` (included in this repo).

```bash
source examples/m/_env.sh
bash examples/m/15_dmc_basins.sh
# smoke knobs: NWALKER=32 NSTEP=100 EQSTEP=20
```

Outputs: `artifacts/nh3_ch3cl/dmc_{react,product}/*.pot`, `*.log`,
`configs_*.traj`. Docs: [`docs/dmc.md`](../../docs/dmc.md).

### ADUMB (PyCHARMM adaptive umbrella)

Yes — the NPZ can drive a PyCHARMM ADUMB job after you have a CGenFF system
(Packmol `AMM1:1,CH3CL:1`, or `07_export_solute_pdb.py` as a lone full-system
PDB / Packmol monomer). There are **no φ/ψ dihedrals** on AMM1/CH3CL; use
RXNCOR + `umbrella rxncor` (same ADUMB path as
`setup/charmm/test/c38test/adumbrxncor.inp`).

| Example | Coordinates | Config / script |
|---------|-------------|-----------------|
| 1D | ξ = \(r_{\mathrm{ClC}}-r_{\mathrm{CN}}\) (`rdif`), [-3, 3] Å, 100 ps | `yaml/adumb_nc_distance.yaml`, `09_adumb_nc_distance.sh` |
| 2D | Cl⋯C + C⋯N (`rcl`, `rcn`) | `yaml/adumb_clc_cn_2d.yaml`, `10_adumb_clc_cn_2d.sh` |
| 1D + solvent | same `rdif` (PBC, 30 Å) | `yaml/adumb_nc_distance_{tip3,acn,dmso}.yaml` (`SOLVATED=1`, optional `SOLVENT=`) |

Requires CHARMM built with **ADUMB** and **ADUMBRXNCOR** (`?ADUMBRXN == 1`).
`scripts/rebuild_charmm_mlpot.sh` adds that pref keyword by default. Without it,
`umbrella rxncor` prints `Unknown umbrella specified` and heat often SIGSEGVs.
The 1D difference window uses `min -3 max 3` — rebuild with the mmml `UM1RXN`
patch in `eadumb.F90` and point `CHARMM_LIB_DIR` at that install (not a stale
PhysNet lib). RXNCOR **NAME** tokens for ADUMB are at most **4 characters**
(`rdif`, `rcl`, `rcn`).

**Production heat:** set `umbrella init` so `nsim * update == heat nstep`
(`nstep ≈ ps_heat * 1000 / dt_fs`). Full notes + results table:
[`docs/examples/nh3-ch3cl-results.md`](../../docs/examples/nh3-ch3cl-results.md)
(ADUMB section).

```bash
# One-time if your libcharmm predates ADUMBRXNCOR / UM1RXN fix:
#   bash scripts/rebuild_charmm_mlpot.sh
source examples/m/_env.sh
rm -rf artifacts/nh3_ch3cl/adumb_nc_distance   # avoid stale next_run / old r_nc lingo

# 100 ps vacuum ADUMB on bond difference ξ=r(ClC)−r(CN) ∈ [-3,3] (NPZ preferred):
USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh

# 2D Cl⋯C + C⋯N adaptive umbrella (smoke ps_heat=0.2):
USE_NPZ_PDB=1 bash examples/m/10_adumb_clc_cn_2d.sh

# Solvated adaptive umbrella (30 Å PBC); combine with USE_NPZ_PDB=1:
SOLVATED=1 bash examples/m/09_adumb_nc_distance.sh
SOLVATED=1 SOLVENT=acn bash examples/m/09_adumb_nc_distance.sh
SOLVATED=1 SOLVENT=dmso bash examples/m/09_adumb_nc_distance.sh
```
If a prior cube Packmol run left monomers ~box-length apart, the script clears
`{output_dir}/.packmol_cache` before launching.

## Pass / fail

| Check | Criterion |
|-------|-----------|
| Evaluate | `artifacts/nh3_ch3cl/evaluate/metrics.json` written; finite MAE/RMSE |
| ASE/JAX-MD smokes | `md_summary.json` with finite `E1`; `md.traj` + `md.xyz` present |
| PyCHARMM `md-system` | DCD under the job `output_dir` (PSF from cluster build) |
| Solute PDB | `solute_amm1_ch3cl.pdb` with 4×AMM1 + 5×CH3CL ATOM lines |
| make-box | `boxes/{acn,tip3,dmso}/model.pdb` + `model.psf` + `box.json` (30 Å default) |
| Mech. embed | campaign exit 0 under `artifacts/nh3_ch3cl/mech_embed_*` |
| ES embed | campaign exit 0 under `artifacts/nh3_ch3cl/es_embed_*` |
| Ewald LR | core `mic_*` / `ewald_*` jobs exit 0; optional libs may SKIP |
| umbrella-sample (gas) | `umbrella_summary.json` + `umbrella_snapshots.npz` under `umbrella_nc_gas/` |
| umbrella-sample (sol) | `engine=hybrid_jaxmd`; snapshots have `energies_unbiased_ev` + `ml_atom_indices` |
| ADUMB 1D | exit 0; lingo has `umbrella rxncor`; ADUMB files under `adumb_nc_distance/` |
| ADUMB 2D | exit 0; lingo has `nrxn 2` + `r_cl`/`r_cn`; ADUMB files under `adumb_clc_cn_2d/` |
| ADUMB solvated | exit 0; ADUMB files under `adumb_nc_distance_{tip3,acn,dmso}/` |
| NEB | exit 0; finite `barrier_kcal_mol`, finite `delta_e_product_kcal_mol` in `neb_summary.json` |
| DMC basins | exit 0; finite average energy in `dmc_{react,product}/*.log` |
| Docs | `docs/examples/nh3-ch3cl-results.md` + PNGs under `docs/images/examples/nh3-ch3cl/` |
