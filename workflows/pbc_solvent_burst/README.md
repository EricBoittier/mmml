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

| Axis | Values |
|------|--------|
| Solvent | DCM, ACO |
| Monomer count | 10, 30, 50, 80, 100 |
| Box | 32 Å cubic PBC |

**10 jobs** with default [config.yaml](config.yaml). Outputs under repo root:

```
artifacts/pbc_solvent_burst/dcm_30/
  campaign.yaml
  campaign_summary.json
  pycharmm_init/
  jaxmd_burst_01/ … jaxmd_burst_05/
  done.txt
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

### Local (limit concurrent GPU jobs)

```bash
snakemake -j2 --resources gpu=1 charmm_slot=1 --keep-going
```

Single-cell smoke:

```bash
snakemake ../../artifacts/pbc_solvent_burst/dcm_10/done.txt -j1 \
  --resources gpu=1 charmm_slot=1
```

### Slurm (up to 10 concurrent GPU jobs)

```bash
nohup bash scripts/snakemake_slurm.sh 10 > snakemake_slurm.log 2>&1 &
```

Or manually:

```bash
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm -j10 \
  --resources gpu=10 charmm_slot=10 --keep-going
```

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
