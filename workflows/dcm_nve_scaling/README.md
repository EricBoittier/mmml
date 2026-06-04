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
- `save_forces_npz: true`, `forces_npz_interval: 1`
- Packmol sphere `R = 18 * (N/60)^(1/3)` Å

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
