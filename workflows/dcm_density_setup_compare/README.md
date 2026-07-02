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
