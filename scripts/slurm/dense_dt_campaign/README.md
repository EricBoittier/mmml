# Denser-box + dt/x64 ensemble campaign

Submitted overnight so results are ready by tomorrow afternoon.

## Why

Prior 30 Å DCM:120 NVT (~0.63 g/cm³) showed bond outliers and E_tot collapse
with PSF angle restraints. NHC invariant drifted only ~2–4 eV / 100 ps.
Next levers (without bond SHAKE yet):

1. **Denser boxes** (L=24 ≈ 1.22 g/cm³, L=26 ≈ 0.96 g/cm³)
2. **dt = 0.5 fs + JAX x64** vs dt = 1 fs float32
3. **NVT / NPT / NVE** comparison

## Runs

| tag | box | ensemble | ps | dt (fs) | x64 |
|---|---|---|---|---|---|
| `L24_nvt_dt1_f32_50ps` | 24 | NVT | 50 | 1.0 | no |
| `L24_nvt_dt05_x64_50ps` | 24 | NVT | 50 | 0.5 | yes |
| `L24_npt_dt1_f32_50ps` | 24 | NPT | 50 | 1.0 | no |
| `L24_npt_dt05_x64_50ps` | 24 | NPT | 50 | 0.5 | yes |
| `L24_nve_dt05_x64_20ps` | 24 | NVE | 20 | 0.5 | yes |
| `L26_nvt_dt1_f32_50ps` | 26 | NVT | 50 | 1.0 | no |
| `L26_npt_dt1_f32_50ps` | 26 | NPT | 50 | 1.0 | no |
| `L30_nvt_dt05_x64_20ps` | 30 (old) | NVT | 20 | 0.5 | yes |

Common: DCM:120, hybrid ML/MM epoch222 + LJ scales, PSF angle restraints,
T=300 K, P=1 atm (NPT), record every 1 ps, GPU jaxmd.

## Where to look

```bash
artifacts/lj_scales/dense_dt_campaign/
  job_ids.txt
  bench.log
  logs/<tag>-<jobid>.{out,err}
  <tag>/bench.log
  <tag>/pbc_*_jaxmd_*.h5          # trajectories
  <tag>/run_meta.txt
artifacts/lj_scales/liquid_dense_L24/box.json
artifacts/lj_scales/liquid_dense_L26/box.json
```

Quick status:

```bash
squeue -u $USER | rg ddc-
tail -50 artifacts/lj_scales/dense_dt_campaign/bench.log
rg -n 'RESULT|H_NHC|nsteps_completed|ERROR' artifacts/lj_scales/dense_dt_campaign/*/bench.log
```

## Scripts

- `scripts/slurm/dense_dt_campaign/submit_all.sh`
- `scripts/slurm/dense_dt_campaign/run_one.sh`
- `scripts/slurm/dense_dt_campaign/sbatch_one.sh`
- `scripts/slurm/dense_dt_campaign/monitor_and_progress.sh` — status + `--react` remediation
- `scripts/slurm/dense_dt_campaign/install_monitor_cron.sh` — every **15 min** cron
- `scripts/slurm/dense_dt_campaign/plot_passed_runs.py` — thermo / RDF / box / bond plots

```bash
bash scripts/slurm/dense_dt_campaign/install_monitor_cron.sh
bash scripts/slurm/dense_dt_campaign/monitor_and_progress.sh --react
uv run python scripts/slurm/dense_dt_campaign/plot_passed_runs.py
```

Live status: `artifacts/lj_scales/dense_dt_campaign/STATUS.md`  
Plots: `artifacts/lj_scales/dense_dt_campaign/plots/`

Note: `sbatch_one.sh` must use `SLURM_SUBMIT_DIR` as repo ROOT (Slurm copies the
script under `/var/spool`, so `BASH_SOURCE` is wrong in the allocation).

## Manuscript link

Supports draft Results §§7–8 (conservation + DCM liquid density) in
`docs/manuscripts/condensed-phase-hybrid-mlmm/`. Dense NPT/NVT at L=24/26 is the
bridge from the sparse L=30 (~0.63 g/cm³) cliff toward bulk ρ≈1.33 before bond
SHAKE / full `pbc_liquid_density_dyn` production tables.
