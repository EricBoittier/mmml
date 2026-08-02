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

### Lever-2 handoff (overbinding)

Epoch222 was trained at `mm_switch_on=8`. Deploy now defaults to **soft** lever-2:

| `DDC_HANDOFF` | `mm_switch_on` | soft-well median | contact ray-min | Use |
|---|---:|---:|---:|---|
| `soft` (default) | 5.0 | ≈ −4.4 kcal | still ≈ −30 | liquid / droplet mitigation |
| `contact` | 3.5 | ≈ −1.1 (underbinds) | ≈ −1.1 | diagnostic only |

```bash
# default soft lever-2
bash scripts/slurm/dense_dt_campaign/submit_all.sh
# contact-ray diagnostic (not for production density)
DDC_HANDOFF=contact bash scripts/slurm/dense_dt_campaign/sbatch_one.sh ...
```

**Contact rays:** deploy-only `on=3.5` kills the −30 kcal wells but flattens the
soft well. The real fix is a GPU retrain at `mm_switch_on=5` (matching deploy)
so the ML local interaction unlearns contact overbinding under the new taper.
See `docs/images/dense-dt-campaign/overbind_ablation/`.

### Retrain at `on=5` (warm-start epoch222)

Config: `examples/hybrid_mm_charges/train_fixed_lj_scales_on5.yaml`  
Tag: `hybrid_mm_lever2_on5_ft` → `artifacts/lj_scales/ckpts/`

```bash
mkdir -p artifacts/lj_scales/dense_dt_campaign/logs
bash scripts/slurm/dense_dt_campaign/submit_train_lever2_on5.sh
# or: sbatch scripts/slurm/dense_dt_campaign/sbatch_train_lever2_on5.sh
```

Defaults: 50 epochs, batch 64, `n_train=32000` / `n_valid=5950`, exclusive GPU
node, hard-fail unless JAX sees `CudaDevice` (avoids silent CPU training).
Overrides: `DDC_ON5_EPOCHS`, `DDC_ON5_BATCH`, `DDC_ON5_TAG`, `DDC_ON5_CKPT`,
`DDC_ON5_DATA`. After train, re-run contact-ok dimer scans /
`ablate_overbind.py` on the new best ckpt + sidecar before swapping campaign
`CKPT`/`SIDECAR` in `run_one.sh`.

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

- `scripts/slurm/dense_dt_campaign/plot_dimer_profiles.py` — DCM–DCM 1D profiles (contact-ok: `dmin ≥ 2 Å`)
- `scripts/slurm/dense_dt_campaign/dimer_scan_contacts.py` — annotate `dmin_A` / clash-filtered summary
- `scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py` — POV stills (skips clashes)
