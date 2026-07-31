# Parallel hybrid umbrella windows (GPU / Snakemake)

One Slurm GPU job per umbrella window for solvated mechanical embedding
(`hybrid_jaxmd`: ML solute + MM solvent). Uses `--windows N --resume` so each
job writes `output_dir/windows/wXXX.npz`; `assemble` + `umbrella-mbar` run after
all windows finish.

## Layout

```text
make_box  →  window[{000..N}]  →  assemble  →  mbar
                 (parallel GPU)
```

Artifacts (default ACN prod):

```text
artifacts/nh3_ch3cl/boxes/{solvent}/model.{psf,pdb}
artifacts/nh3_ch3cl/umbrella_nc_{solvent}_prod/
  windows/wXXX.npz
  umbrella_snapshots.npz
  umbrella_summary.json
  mbar/status.json
  logs/window_wXXX.log
```

## Prerequisites

```bash
cd ~/mmml
ls examples/m/model_ext.json
uv sync --extra gpu
uv sync --extra mbar
export CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-$HOME/.cache/mmml-charmm-build/tier_56000000_nodomdec/lib}
```

## Dry-run

```bash
cd workflows/hybrid_umbrella_windows
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2 -n
```

## Studix GPU queue

Submit from the **login node** (do not run JAX on login):

```bash
cd workflows/hybrid_umbrella_windows

# Smoke: 3 TIP3 windows, up to 3 concurrent GPUs
MMML_WORKFLOW_CONFIG=config.smoke.yaml \
  nohup bash scripts/snakemake_slurm.sh 3 > snakemake_gpu.log 2>&1 &

# ACN production (default config.yaml): 30 windows, 8 concurrent
nohup bash scripts/snakemake_slurm.sh 8 > snakemake_gpu.log 2>&1 &

# TIP3 production
MMML_WORKFLOW_CONFIG=config.tip3.yaml \
  nohup bash scripts/snakemake_slurm.sh 8 > snakemake_gpu.log 2>&1 &

tail -f snakemake_gpu.log
squeue -u "$USER"
```

From `examples/m`:

```bash
SOLVENT=acn JOBS=8 bash examples/m/15_umbrella_snakemake.sh
```

## Resume / fill holes

Snakemake only re-runs missing `windows/wXXX.npz`. Failed MD still writes a
checkpoint (`status=failed`), so that window is considered done; re-run a hole
by deleting its NPZ:

```bash
rm artifacts/nh3_ch3cl/umbrella_nc_acn_prod/windows/w019.npz
bash scripts/snakemake_slurm.sh 8
```

Or force a subset via config:

```yaml
window_ids: [19, 20, 21]
```

## Tuning

| Key | Meaning |
|-----|---------|
| `slurm.max_jobs` | Concurrent GPU window jobs (`-j`) |
| `slurm.nodelist` | Pin nodes, e.g. `gpu08,gpu09` |
| `slurm.runtime_min_window` | Walltime minutes per window |
| `n_windows` | Must match YAML schedule (or override consistently) |
| `run_mbar` | Set `false` to stop after assemble |
