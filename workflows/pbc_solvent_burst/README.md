# PBC DCM / ACO burst campaign workflow

Snakemake workflow for **DCM** and **ACO** clusters in a **32 Å** cubic PBC box at sizes **10, 30, 50, 80, 100**. Each matrix cell runs one in-process `mmml md-system --run-all` campaign:

1. **PyCHARMM init** — MLpot mini + gentle segmented heat (overlap rescue + bonded-MM repair)
2. **PyCHARMM equi** — 5 × 10 ps NPT equilibration segments (50 ps total)
3. **JAX-MD bursts** — 5 × 200 ps NVT (1 ns total), with PBC FIRE at handoff and CHARMM overlap rescue during dynamics

Sibling workflows: [dcm_heat_scaling](../dcm_heat_scaling/) (long heat-only screening), [dcm5_md_benchmark](../dcm5_md_benchmark/) (cross-backend smoke).

## Do not commit run outputs

`artifacts/pbc_solvent_burst/`, `results/`, and `.snakemake/` are gitignored.

## Prerequisites

```bash
# Default checkpoint is set in config.yaml (DESdimers_params.json).
# Optional override when config uses ${MMML_CKPT}:
# export MMML_CKPT=/path/to/DESdimers_params.json
export JAX_ENABLE_X64=1   # optional; job_shell defaults to 1
```

- GPU JAX (`uv sync --extra gpu`) or cluster env with `jax` + `mmml`
- `packmol` on PATH (cube placement inside 32 Å box)
- `snakemake` and [`snakemake-executor-plugin-slurm`](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) for cluster submission

```bash
uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake --version
```

## Job matrix

| Axis | Values (config keys) |
|------|----------------------|
| Solvent | `solvents` |
| Monomer count | `cluster_sizes` |
| Temperature (K) | `temperatures` (list) |
| Box (Å) | `box_sizes` (list) |

Run tag: `{solvent}_{n}` when there is one temperature and one box (default). When sweeping `temperatures` or `box_sizes`, tags include T/L: `dcm_10_t300_l32`.

Outputs:

```
artifacts/pbc_solvent_burst/dcm_30_t320_l28/
  campaign.yaml
  pycharmm_init/pretreat/ …
  jaxmd_burst_01/ …
  done.txt
```

Set `output_root` to an absolute path outside the repo if you prefer (e.g. `/mmhome/.../runs/pbc_burst`).

## Cleanup strategy (`cleanup_strategy` in config.yaml)

When geometry or handoff quality breaks, mmml already runs a hybrid recovery ladder. The workflow maps YAML to those hooks:

| Step | YAML block | When it runs |
|------|------------|--------------|
| CHARMM MM heat/equi/prod | `charmm_mm.pretreat_on_pycharmm` | Before MLpot on each PyCHARMM leg (preventive) |
| CHARMM overlap SD/ABNR | `charmm_mm.overlap_rescue_*` | During PyCHARMM dynamics overlap rescue |
| MLpot bonded mini + rescue | `mlpot.*` | After mini/heat strain; overlap rescue on PyCHARMM |
| JAX PBC FIRE + CHARMM rescue | `jaxmd_pbc.*` | Handoff quality gate; JAX-MD overlap → CHARMM |

Enable pretreat for your A/B test:

```yaml
cleanup_strategy:
  charmm_mm:
    pretreat_on_pycharmm: true
    ps_heat: 30.0
    ps_equi: 10.0
    ps_prod: 5.0
```

Sweep temperature / box:

```yaml
temperatures: [280, 300, 320]
box_sizes: [28, 32, 36]
```

### Density warning

N=80–100 in a 32 Å cube is **extremely dense** (400–1000 atoms). Expect frequent overlap rescue; Packmol may fail at the largest sizes. Sizes listed in `optional_sizes: [100]` mark the final JAX-MD burst as `optional` in the campaign (failure does not abort earlier legs when using `--keep-going` at the Snakemake level; within-campaign abort still stops that cell).

Tune or drop large sizes in `config.yaml` if placement fails.

## Campaign timing (defaults)

| Leg | Count | ps each | Total |
|-----|-------|---------|-------|
| Init heat | 1 | 30 | 30 ps |
| PyCHARMM equi | 5 | 10 | 50 ps |
| JAX-MD NVT | 5 | 200 | 1000 ps (1 ns) |

Repair defaults: `dynamics_overlap_action: rescue` (PyCHARMM), bonded-MM after mini/heat, JAX-MD overlap CHARMM rescue + velocity rethermalization + `handoff_quality_gate`.

## Run

```bash
cd workflows/pbc_solvent_burst
bash scripts/preflight.sh
snakemake -n
```

### Local (match GPU count on your machine)

```bash
# 4 GPUs → run 4 matrix cells at once
snakemake -j4 --resources gpu=4 charmm_slot=4 --keep-going
```

Single-cell smoke:

```bash
snakemake ../../artifacts/pbc_solvent_burst/dcm_10/done.txt -j1 \
  --resources gpu=1 charmm_slot=1

# or directly:
bash scripts/job_shell.sh dcm_10
```

### Slurm (max throughput)

Set **`slurm_max_concurrent`** in [config.yaml](config.yaml) to the **total number of GPUs** you want to use across `slurm_gpu_nodes` (default **10** = all matrix cells at once). The launcher sets **`gpu` and `charmm_slot` pools to the same value** so Snakemake does not serialize jobs.

```bash
# uses slurm_max_concurrent from config.yaml (default 10)
nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &

# or override for this launch only (e.g. 8 GPUs on gpu08+gpu09)
nohup bash scripts/snakemake_slurm.sh 8 > snakemake_slurm.log 2>&1 &
```

Or manually:

```bash
N=10   # GPUs available
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm -j"$N" \
  --resources "gpu=${N}" "charmm_slot=${N}" --keep-going
```

**Do not** use `--resources gpu=10 charmm_slot=1` for throughput — `charmm_slot=1` caps the workflow to **one** Snakemake job at a time even if GPUs are idle.

Slurm may still queue jobs if `slurm_max_concurrent` exceeds free GPUs; that is normal. Multiple one-GPU jobs on the same node share CPU/RAM during PyCHARMM legs but use separate GPUs during JAX-MD bursts.

Default runtime: **48 h** per job (`slurm_runtime_min: 2880`). Adjust in `config.yaml` if 1 ns JAX-MD per cell needs more wall time.

## Resume

Each cell uses `mmml md-system --resume-campaign`, which skips legs whose output dirs already have valid handoffs. Re-run a single cell:

```bash
bash scripts/job_shell.sh DCM 30
```

Or regenerate and dry-run one campaign YAML:

```bash
python3 scripts/campaign_lib.py   # not a CLI — use run_job or:
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from campaign_lib import load_config, write_campaign_yaml
cfg = load_config(Path('config.yaml'))
print(write_campaign_yaml(cfg, 'DCM', 30))
"
```

## Status

```bash
bash scripts/status.sh
# writes workflows/pbc_solvent_burst/results/status.csv
```

## Configuration

Edit [config.yaml](config.yaml) for heat segments, burst length, overlap thresholds, Slurm partition/nodelist, and `optional_sizes`.

Campaign YAML is generated per cell by [scripts/campaign_lib.py](scripts/campaign_lib.py) — do not hand-edit `artifacts/.../campaign.yaml` unless debugging a single rerun.

## Tests

```bash
uv run pytest tests/unit/test_pbc_solvent_burst_campaign.py -q
```
