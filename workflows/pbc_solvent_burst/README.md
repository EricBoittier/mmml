# PBC DCM / ACO burst campaign workflow

Snakemake workflow for **DCM** and **ACO** in cubic PBC boxes. Each matrix cell runs one in-process `mmml md-system --run-all` campaign:

1. **PyCHARMM init** — MLpot mini + gentle segmented heat (overlap rescue + bonded-MM repair)
2. **PyCHARMM equi** — 5 × 10 ps NPT equilibration segments (50 ps total)
3. **JAX-MD bursts** — 5 × 200 ps NVT (1 ns total), with PBC FIRE at handoff and CHARMM overlap rescue during dynamics

Sibling workflows: [dcm_heat_scaling](../dcm_heat_scaling/) (long heat-only screening), [dcm5_md_benchmark](../dcm5_md_benchmark/) (cross-backend smoke), [pbc_liquid_density_dyn](../pbc_liquid_density_dyn/) (liquid-density PyCHARMM equil + prod).

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
| Bulk density | `bulk_density_fractions` — N = fraction × 298 K liquid count per solvent/box |
| Legacy fixed N | `cluster_sizes` — use instead of `bulk_density_fractions` (mutually exclusive) |
| Temperature (K) | `temperatures` (list) |
| Box (Å) | `box_sizes` (list) |

**Bulk reference** (monomers at 100% liquid density):

| L (Å) | DCM N | ACO N |
|-------|-------|-------|
| 28 | 206 | 178 |
| 32 | 308 | 266 |
| 36 | 439 | 379 |

Default fractions `[0.5, 0.75, 1.0]` → e.g. `dcm_103_t200_l28` (50% bulk DCM), `dcm_206_t200_l28` (100%).

Run `scripts/preflight.sh` to print the full table for configured `box_sizes`.

Run tag: `{solvent}_{n}` when there is one temperature and one box (default). When sweeping `temperatures` or `box_sizes`, tags include T/L: `dcm_154_t300_l32`.

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

Pretreat equi/prod use **Hoover NVT at the matrix `box_size`** (fixed L), not NPT — so the cell stays at 32 Å (or whatever L you chose) and matches JAX-MD PBC.

Sweep temperature / box:

```yaml
temperatures: [280, 300, 320]
box_sizes: [28, 32, 36]
```

### Density warning

N=80–100 in a 32 Å cube is **extremely dense** when using legacy fixed `cluster_sizes`. With **bulk-density** sizing, 0.5× liquid (~154 DCM in L=32) is the moderate tier; 1.0× (~308) is full liquid and may stress Packmol or heat — those cells mark the final JAX burst `optional` via `optional_bulk_fractions: [1.0]`.

Tune or drop large sizes in `config.yaml` if placement fails.

### MLpot heat stability (large N, cold T)

Early heat abort (`restart step ~200 < nstep`, `echeck` stop, wild CPT piston) on **N≥80** / low-T cells is distinct from extent-recovery bugs. Defaults mitigate this:

| Setting | Purpose |
|---------|---------|
| `heat_thermostat: hoover` | MLpot heat uses Hoover CPT (required with pretreat + overlap chunks; campaign coerces `scale`→`hoover` when pretreat is on) |
| `cleanup_strategy.mlpot.no_echeck_heat: true` | Disables CHARMM ECHECK during heat only; equi/prod still use scaled `--echeck` |
| `dynamics_overlap_check_interval: 250` | Extent/overlap checks every ~250 steps **inside** each heat segment (not only at segment end) |
| `dcd_nsavc: 100` | DCD save interval; must be `<` overlap interval so chunk DCDs get frames |

For legacy Hoover heat that checks only at segment boundaries, pass `--heat-overlap-segment-boundary-only`.

For heat-only screening with global echeck off, see [dcm_heat_scaling](../dcm_heat_scaling/) (`no_echeck: true`).

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

### Local (workstation with PyCHARMM + OpenCL on the same machine)

On **pc-studix login nodes**, PyCHARMM fails with `libOpenCL.so.1: cannot open shared object file` — use the **Slurm** section below instead.

```bash
# 4 GPUs → run 4 matrix cells at once (only where libOpenCL is available)
snakemake -j4 --resources gpu=4 charmm_slot=4 --keep-going
```

Single-cell smoke on a **GPU node** (full tag when sweeping T/box):

```bash
# Slurm (recommended on cluster):
bash scripts/snakemake_slurm.sh 1 ../../artifacts/pbc_solvent_burst/dcm_10_t300_l32/done.txt

# Or srun one cell:
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 \
  bash scripts/job_shell.sh dcm_10_t300_l32
```

### Slurm (max throughput)

Set **`slurm_max_concurrent_fast`** / **`slurm_max_concurrent_slow`** (or legacy **`slurm_max_concurrent`**) so Snakemake job pools match GPU counts. The launcher sets **`gpu_fast`**, **`gpu_slow`**, and **`charmm_slot`** automatically.

**Tiered scheduling** (default in `config.yaml`): matrix cells with **`N ≤ slurm_small_cluster_max_n`** (default 30) go to **`slurm_gpu_nodes_slow`** (3080); larger clusters go to **`slurm_gpu_nodes_fast`** (5090/4090/3090). Disable by removing `slurm_gpu_nodes_slow`.

```bash
bash scripts/preflight.sh   # prints tier pools and N threshold

# One driver only — stop old ones before relaunching:
bash scripts/stop_snakemake.sh
snakemake --profile profiles/slurm --unlock

nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &
pgrep -af 'snakemake --profile profiles/slurm'   # expect exactly one uv + one python
```

Or manually:

```bash
N=10   # GPUs available
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm -j"$N" \
  --resources gpu="$N" charmm_slot="$N" --keep-going
```

**Do not** use `--resources gpu=10 charmm_slot=1` for throughput — `charmm_slot=1` caps the workflow to **one** Snakemake job at a time even if GPUs are idle.

Slurm may still queue jobs if `slurm_max_concurrent` exceeds free GPUs; that is normal. Multiple one-GPU jobs on the same node share CPU/RAM during PyCHARMM legs but use separate GPUs during JAX-MD bursts.

Default runtime: **48 h** per job (`slurm_runtime_min: 2880`). Adjust in `config.yaml` if 1 ns JAX-MD per cell needs more wall time.

**Slurm nodelist:** per-job `--nodelist` comes from the tier (`slurm_gpu_nodes_fast` / `slurm_gpu_nodes_slow`). Omit both slow and fast lists and set only `slurm_gpu_nodes` for flat scheduling. Restart Snakemake after changing — jobs already submitted keep the old `--nodelist`.

## Resume

Each cell uses `mmml md-system --resume`, which skips legs whose output dirs already have valid handoffs. Re-run a single cell:

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

Health dashboard for the full matrix (leg progress, DYNA energy/T from `stdout.log`, restart steps):

```bash
bash scripts/status.sh
bash scripts/status.sh -v                      # last DYNA> lines per run
bash scripts/status.sh --failed -v             # failures only
bash scripts/status.sh --tag dcm_154_t150_l32  # one-cell deep dive
bash scripts/status.sh --plot-dir results/plots  # PNG time series (needs matplotlib)
bash scripts/status.sh --json results/status.json
```

Writes `results/status.csv` with extended columns (`health`, `dyna_n_frames`, `temperature_last_K`, …).

Post-mortem for a failed Slurm cell (errors, tail log, restarts):

```bash
bash scripts/diag_cell.sh dcm_154_t150_l32
```

## Configuration

Edit [config.yaml](config.yaml) for heat segments, burst length, overlap thresholds, Slurm partition/nodelist, and `optional_sizes`.

Campaign YAML is generated per cell by [scripts/campaign_lib.py](scripts/campaign_lib.py) — do not hand-edit `artifacts/.../campaign.yaml` unless debugging a single rerun.

## Tests

```bash
uv run pytest tests/unit/test_pbc_solvent_burst_campaign.py -q
```
