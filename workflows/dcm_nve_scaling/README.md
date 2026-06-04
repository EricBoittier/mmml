# DCM NVE scaling (PyCHARMM MLpot)

Free-space **NVE** runs for **DCM:5 … DCM:10** with per-step CHARMM print, DCD frames,
optional force NPZ dumps, post-mini audits, and per-monomer COM displacement analysis.

Sibling to [dcm5_md_benchmark](../dcm5_md_benchmark/) (fixed DCM:5, multi-backend smoke).

## Do not commit run outputs

`results/` and `.snakemake/` are gitignored.

## Prerequisites

- `export MMML_CKPT=/path/to/dcm_physnet_ckpt`
- GPU JAX (`uv sync --extra gpu`)
- OpenMPI + rebuilt `libcharmm.so` for large clusters if you extend beyond N=10
- `packmol` on PATH
- `snakemake`

## Config highlights

From [config.yaml](config.yaml):

- `dcd_nsavc: 1`, `dyn_nprint: 1`, `nprint: 1` (every integration step)
- `nve_boltzmann_temp: 30` (gentle velocity draw before NVE; `--temperature` is for NVT stages only)
- `mini_nstep: 1000`, `packmol_reference_r: 20`, `packmol_tolerance: 1.2` (looser initial cluster)
- `forces_npz_interval: 500` (DCD still every step via `dcd_nsavc: 1`)
- `dynamics_overlap_check_interval: 80000` (single NVE chunk at 20 ps / 0.25 fs; no scratch restart handoff)
- `no_echeck: true` for the 20 ps NVE leg (in-run echeck stops ML USER clusters within ~1k steps; `run_job.py` still validates DCD length)
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

Outputs per size under `results/dcm_N_nve/`:

- `nve_dcm_N.dcd`, `mini_full_mlpot_dcm_N.crd`
- `audit.json`, `com_analysis.npz`, optional `forces.npz`
- Aggregated `results/scaling_summary.csv`, `scaling_report.md`

## Analysis scripts (repo root)

- `scripts/audit_mlpot_cluster.py`
- `scripts/analyze_monomer_com_dcd.py`
