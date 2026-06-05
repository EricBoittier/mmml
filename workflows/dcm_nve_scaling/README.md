# DCM NVE scaling (PyCHARMM MLpot)

Free-space **NVE** runs for **DCM:3** and **DCM:5 … DCM:10** with per-step CHARMM print, DCD frames,
optional force NPZ dumps, post-mini audits, and per-monomer COM displacement analysis.

Sibling to [dcm5_md_benchmark](../dcm5_md_benchmark/) (fixed DCM:5, multi-backend smoke).

## Do not commit run outputs

`results/` and `.snakemake/` are gitignored.

## Prerequisites

- `export MMML_CKPT=/path/to/dcm_physnet_ckpt`
- GPU JAX (`uv sync --extra gpu`) **or** an activated conda/micromamba env with `jax` + `mmml` installed
- Optional: `export MMML_PYTHON=/path/to/python` if Snakemake does not inherit your env (workflow scripts prefer `$CONDA_PREFIX/bin/python` over bare `python3` on PATH)
- OpenMPI + rebuilt `libcharmm.so` for large clusters if you extend beyond N=10
- `packmol` on PATH
- `snakemake`

## Config highlights

From [config.yaml](config.yaml):

- `dcd_nsavc: 1`, `dyn_nprint: 1`, `nprint: 1` (every integration step)
- `nve_boltzmann_temp: 0.2` (very low initial KE before NVE; raise cautiously if clusters are too cold)
- `ps_nve: 0.2` (800-step screening leg; increase once COM QC is stable)
- `nve_inbfrq_values: [-1, 1, 10, 50]` — one run per value under `results/dcm_N_nve/inbfrq_<slug>/`
- `pre_nve_charmm_update: true` — CHARMM `ENER`+`UPDATE` after mini (lists were frozen at `inbfrq=0`)
- `mini_nstep: 2000`, `bonded_mm_mini: true` (bonded-only SD after mini if ANGL/GRMS spike)
- `packmol_reference_r: 20`, `packmol_tolerance: 1.2` (looser initial cluster)
- `forces_npz_interval: 500` (DCD still every step via `dcd_nsavc: 1`)
- `dynamics_overlap_check_interval: 800` (single NVE chunk; matches `ps_nve` / `dt_fs`)
- `no_echeck: true` for the 0.5 ps NVE leg (in-run echeck stops ML USER clusters within ~1k steps; `run_job.py` still validates DCD length)
- `save_forces_npz: true`, `forces_npz_interval: 1`
- Packmol sphere `R = 18 * (N/60)^(1/3)` Å
- **Free-space ML dimers:** `max_active_dimers = N(N−1)/2` (every unique pair evaluated each step; not the PBC `max(1000, 6N)` cap). Unset `MMML_MLPOT_MAX_ACTIVE_DIMERS` unless you intentionally override this.

## Run

```bash
export MMML_CKPT=/path/to/your/dcm_ckpt
cd workflows/dcm_nve_scaling
bash scripts/preflight.sh
snakemake -n
snakemake -j3 --resources gpu=1 mpi=1 --keep-going
```

Single size:

```bash
snakemake results/dcm_7_nve/done.txt --cores 1
```

Outputs per size and `inbfrq` under `results/dcm_N_nve/inbfrq_<slug>/`:

- `nve_dcm_N.dcd`, `mini_full_mlpot_dcm_N.crd`, `stdout.log`
- `audit.json`, `com_analysis.npz`, optional `forces.npz`
- Aggregated `results/scaling_summary.csv` (includes `inbfrq` column), `scaling_report.md`

Pick the best `inbfrq` from the CSV (`status=pass`, low `outlier_ratio`), then optionally re-run production with that value only (set `nve_inbfrq_values: [-1]`).

## Analysis scripts (repo root)

- `scripts/audit_mlpot_cluster.py`
- `scripts/analyze_monomer_com_dcd.py` (cluster COM drift + MSD, internal RMSD, monomer outlier ratio)

`com_analysis` uses `com_analysis.no_fail: true` in config so unstable short NVE runs still write
`com_analysis.npz` and Snakemake can finish `collect` (status `fail` in CSV when checks fail).
Re-run analysis only:

```bash
python ../../scripts/analyze_monomer_com_dcd.py \
  --dcd results/dcm_5_nve/nve_dcm_5.dcd --n-monomers 5 --atoms-per-monomer 5 \
  --no-fail -o results/dcm_5_nve/com_analysis.npz
```
