# DCM density × setup comparison

Snakemake workflow that compares **minimization / prep setups** for DCM PBC clusters at the **bulk-density fractions and box sizes** used in [pbc_solvent_burst](../pbc_solvent_burst/). Each matrix cell runs a **mini-only** `pycharmm_mini` leg (no heat, no JAX bursts) to see which prep stack survives Packmol placement at that density.

## Goal

Answer: *at 0.25× and 0.5× bulk liquid density (L = 28–32 Å), which prep path reaches a valid mini handoff?*

| Setup | Description |
|-------|-------------|
| `minimal` | Packmol → MLpot CHARMM SD only |
| `calculator_pre_sd` | ASE hybrid FIRE/BFGS before CHARMM SD |
| `liquid_prep_dense` | `liquid_prep` + `density_prep_ladder` |
| `burst_hybrid` | [pbc_solvent_burst](../pbc_solvent_burst/) cleanup ladder (pretreat + rescue) |
| `resilient` | liquid prep + calculator pre-SD + resilient cleanup |

## Do not commit run outputs

`artifacts/dcm_density_setup_compare/`, `results/`, `.snakemake/` are gitignored.

## Prerequisites

```bash
export MMML_CKPT=/path/to/DESdimers_params.json
export JAX_ENABLE_X64=1
```

GPU node with PyCHARMM + OpenCL, `packmol`, `snakemake`.

## Matrix (default `config.yaml`)

| Axis | Values |
|------|--------|
| Setup | 5 variants (see table above) |
| Solvent | DCM only |
| Bulk density | `0.25`, `0.5` (same as burst default fractions) |
| Temperature | 300 K |
| Box (Å) | 28, 30, 32 |

**N at 100% liquid (298 K, reference):**

| L (Å) | DCM N_bulk |
|-------|------------|
| 28 | 206 |
| 30 | 251 |
| 32 | 308 |

Example run tag: `liquid_prep_dense_dcm_77_t300_l32` (0.25× bulk DCM in L=32).

Default matrix size: **30 cells** (5 setups × 2 density fractions × 3 boxes).

## Campaign legs (per cell)

| Leg | Stages |
|-----|--------|
| `pycharmm_mini` | `mini` only |

Outputs per cell:

```
artifacts/dcm_density_setup_compare/minimal_dcm_52_t300_l28/
  campaign.yaml
  campaign_summary.json
  pycharmm_mini/handoff/state.npz
  done.txt
```

## Run

```bash
cd workflows/dcm_density_setup_compare
bash scripts/preflight.sh
snakemake -n
```

### pc-studix (login node → Slurm)

PyCHARMM needs **OpenCL on GPU compute nodes** — do not run `job_shell.sh` on the login node (`libOpenCL.so.1` missing). Submit via Snakemake + Slurm from the login node:

```bash
cd /mmhome/boittier/home/mmml/workflows/dcm_density_setup_compare   # adjust path
export MMML_CKPT=/mmhome/boittier/home/mmml/examples/ckpts_json/DESdimers_params.json
export JAX_ENABLE_X64=1

bash scripts/preflight.sh

# One driver only — stop old ones before relaunching:
bash scripts/stop_snakemake.sh 2>/dev/null || true
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm --unlock

# Full matrix (30 cells, tiered 3080/5090 pools from config.yaml):
nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &
tail -f snakemake_slurm.log

# Monitor driver:
pgrep -af 'snakemake --profile profiles/slurm'
```

Single-cell smoke (still from login — Snakemake submits one GPU jobstep):

```bash
bash scripts/snakemake_slurm.sh 1 \
  ../../artifacts/dcm_density_setup_compare/minimal_dcm_52_t300_l28/done.txt
```

Or direct `srun` on a GPU node:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 \
  bash scripts/job_shell.sh minimal_dcm_52_t300_l28
```

Requires `snakemake` + `snakemake-executor-plugin-slurm` (Snakemake 8+):

```bash
uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake --version
```

### Local dry-run (workstation with OpenCL, not pc-studix login)

```bash
snakemake --profile profiles/local -n
```

## Troubleshooting failed cells

Snakemake marks a cell failed when `done.txt` is missing. Check the per-cell log:

```bash
TAG=liquid_prep_dense_dcm_127_t300_l30
grep -E 'Packmol failed|pycharmm_mlpot: error|ERROR|failed to converge' \
  ../../artifacts/dcm_density_setup_compare/${TAG}/stdout.log | tail -20
```

Common causes:

| Symptom | Likely cause |
|---------|----------------|
| `Packmol failed` / `failed to converge` | Inner cube too small vs N (fixed: `packmol_box_padding: 1.0` in config) |
| `libOpenCL.so.1 not found` | Job ran on login node — use Snakemake Slurm profile |
| `MMML_CKPT is not set` | Export `MMML_CKPT` before `snakemake` (passed via `envvars` in Slurm profile) |
| `Checkpoint not found` | Wrong path in `MMML_CKPT` |

Some matrix cells are **expected** to fail mini (overlap / prep stack comparison). That is not a workflow bug — use `campaign_summary.json` per cell to compare which setups reached handoff.

## Prep-parameter sweep (anchor cell × variants)

To compare **what actually moves prep** (timestep, Packmol tolerance, cutoffs, MM on/off) without the full 108-cell matrix, use `config.prep_sweep.yaml`:

| Setting | Role |
|---------|------|
| `prep_sweep.anchor` | Single cell (default: resilient, 0.25× bulk, T=50 K, L=28 Å) |
| `prep_sweep.variants` | Named one-at-a-time overrides vs shared baseline |
| `prep_sweep.stages` | `mini` (default) or `mini,heat` |
| Tag suffix | `_sw_{variant}` e.g. `resilient_dcm_52_t50_l28_sw_pmtol30` |

Default variants (stage 1): `baseline`, `dt050`, `pmtol25/30`, `spacing50/70`, `cut_tight/wide`, `mm_bonded_on`, `mm_pretreat_off`, `mm_vdw_off`.

**Stage 2** (`config.prep_sweep.yaml` as committed): anchor **vdw_off + spacing50**, `stages: mini,heat`, variants `baseline`, `dt050`/`dt010`/`dt015`, `ovlp50`/`ovlp25`, `dcd50`/`dcd25`, `inbfrq25`/`inbfrq10`, `dt050_ovlp50`. Tags include `_ht_bussi_`, e.g. `resilient_dcm_52_t50_l28_ht_bussi_sw_dt010`.

```bash
cd workflows/dcm_density_setup_compare
export MMML_CKPT=/path/to/DESdimers_params.json
bash scripts/preflight.sh   # uses config.yaml unless MMML_WORKFLOW_CONFIG is set

# Prep sweep (11 jobs) — must set MMML_WORKFLOW_CONFIG for driver AND Slurm jobs:
MMML_WORKFLOW_CONFIG=config.prep_sweep.yaml bash scripts/preflight.sh
snakemake --configfile config.prep_sweep.yaml --profile profiles/slurm -n
nohup bash scripts/snakemake_prep_sweep.sh > snakemake_prep_sweep.log 2>&1 &
bash scripts/collect_prep_sweep.sh
# -> results/prep_sweep_summary.csv
```

**Important:** Do not pass only `--configfile` to Snakemake without `MMML_WORKFLOW_CONFIG` — compute jobs would still read `config.yaml` and fail on `_sw_*` tags. Use `snakemake_prep_sweep.sh` or export `MMML_WORKFLOW_CONFIG` before launching.

Add your own variant under `prep_sweep.variants` (lowercase id, mapping of md-system keys). Set `prep_sweep.stages: mini,heat` and `anchor.heat_thermostat: bussi` to test heat/overlap on the same anchor.

## Customize

- Add/remove setups in `config.yaml` → `setups:` (ids defined in `scripts/setup_variants.py`).
- Match burst temperatures: set `temperatures: [50.0, 100.0, 150.0]`.
- Add `bulk_density_fractions: [0.75, 1.0]` for denser cells (watch Packmol / overlap).
- Skip hard cells: `exclude_run_tags: [minimal_dcm_206_t300_l28]`.

## Relation to pbc_solvent_burst

| | pbc_solvent_burst | dcm_density_setup_compare |
|--|-------------------|---------------------------|
| Solvents | DCM, ACO | **DCM only** |
| Density | 0.25–0.5 (default) | **Same fractions / boxes** |
| Axis | T sweep | **Setup variant** |
| Dynamics | mini+heat + JAX bursts | **mini-only smoke** |

## Tests

```bash
uv run pytest tests/unit/test_dcm_density_setup_compare_campaign.py -q
```
