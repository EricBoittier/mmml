# PBC liquid-density cluster dynamics (persistent PyCHARMM)

Snakemake workflow to equilibrate and run **production MD** on molecular clusters at **bulk liquid density** in cubic PBC. Unlike [pbc_solvent_burst](../pbc_solvent_burst/) (JAX-MD burst alternation), this workflow stays on **PyCHARMM** for the full trajectory and leans on mmml's resilient prep tools.

## Goal

Reach stable **liquid-density** configurations and accumulate **persistent dynamics** (segmented NPT equil + NPT production) with:

- `liquid_prep` + `density_prep_ladder` (Packmol / MC / geometry / CHARMM / MLpot recovery)
- `cleanup` + overlap rescue + bonded-MM repair
- `mmml warmup-mlpot-jax` (serial JAX cache) → `mmml-charmm-mpirun.sh md-system --run-all --resume`
- Snakemake matrix over solvent × bulk-density fraction × T × box

## Do not commit run outputs

`artifacts/pbc_liquid_density_dyn/`, `results/`, `.snakemake/` are gitignored.

## Prerequisites

```bash
export MMML_CKPT=/path/to/DESdimers_params.json
export JAX_ENABLE_X64=1
```

- GPU node with PyCHARMM + OpenCL
- `packmol`, `snakemake`, `snakemake-executor-plugin-slurm` (cluster)

## Matrix (default `config.yaml`)

| Axis | Values |
|------|--------|
| Solvent | DCM, ACO |
| Bulk density | `bulk_density_fractions: [0.9, 1.0]` |
| Temperature | 300 K (extend via `temperatures`) |
| Box (Å) | 28, 32 |

**N at 100% liquid (298 K):**

| L (Å) | DCM | ACO |
|-------|-----|-----|
| 28 | 206 | 178 |
| 32 | 308 | 266 |

Run tag: `dcm_277_t300_l32` (90% bulk DCM in L=32).

## Campaign legs (per cell)

| Leg | Stages | Default ps |
|-----|--------|------------|
| `pycharmm_init` | prep ladder + mini + heat | 10 ps heat |
| `pycharmm_equi_01` … `05` | NPT equil | 5 × 20 ps = 100 ps |
| `pycharmm_prod_01` … `10` | NPT production | 10 × 50 ps = 500 ps |

All legs use `--resume`; interrupted Slurm jobs continue from the last valid handoff.

Outputs per cell:

```
artifacts/pbc_liquid_density_dyn/dcm_277_t300_l32/
  campaign.yaml
  prep_ladder/journal.json
  pycharmm_init/ …
  pycharmm_prod_10/handoff/state.npz
  done.txt
```

## Run

```bash
cd workflows/pbc_liquid_density_dyn
bash scripts/preflight.sh
snakemake -n
```

### Single cell (gpu09 smoke)

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 \
  bash scripts/job_shell.sh dcm_10_t300_l28
```

### Slurm driver

```bash
nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &
bash scripts/status.sh
```

### pc-bach CPU cluster (`long` partition)

Module stack (example):

```bash
module load gcc/gcc-12.2.0-cmake-3.25.1-openmpi-4.1.4
module load charmm/c47a2-gcc-12.2.0-openmpi-4.1.4
export MMML_CKPT=/path/to/DESdimers_params.json
export CHARMM_HOME=... CHARMM_LIB_DIR=...
```

Uses `profiles/slurm-cpu` (no `--gres=gpu`) and `config.pc-bach.cpu.yaml` (`scheduler: cpu`, `MMML_MLPOT_DEVICE=cpu`):

```bash
cd workflows/pbc_liquid_density_dyn
bash scripts/preflight.sh --config config.pc-bach.cpu.yaml
bash scripts/snakemake_slurm_cpu.sh

# Or explicitly:
MMML_SNAKEMAKE_PROFILE=profiles/slurm-cpu \
MMML_WORKFLOW_CONFIG=config.pc-bach.cpu.yaml \
  bash scripts/snakemake_slurm.sh
```

Single cell on `long`:

```bash
MMML_WORKFLOW_CONFIG=config.pc-bach.cpu.yaml \
  srun --partition=long --cpus-per-task=8 --mem=32G \
  bash scripts/job_shell.sh dcm_10_t300_l28
```

Default concurrency: 17 jobs (`slurm_max_concurrent` = idle nodes on `long`). No `slurm_cpu_nodes` nodelist — Slurm picks any node in the partition.

## Resume

Re-run Snakemake or `job_shell.sh TAG` — `mmml md-system --run-all --resume` skips completed legs. Prep ladder state lives under `prep_ladder/`; campaign summary under `campaign_summary.json`.

## Relation to pbc_solvent_burst

| | pbc_solvent_burst | pbc_liquid_density_dyn |
|--|-------------------|------------------------|
| Density focus | 0.25–1.0 sweep | 0.9–1.0 liquid |
| Dynamics | PyCHARMM equi + **JAX-MD bursts** | **PyCHARMM equi + prod** only |
| Prep | cleanup + pretreat | **density_prep_ladder** + cleanup |
| MPI | auto-rerun in md-system | explicit **mpirun wrapper** + warmup |

## Tests

```bash
uv run pytest tests/unit/test_pbc_liquid_density_dyn_campaign.py -q
```
