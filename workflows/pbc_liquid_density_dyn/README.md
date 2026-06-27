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

Three-step setup on **pc-bach** (OpenMPI **4.1.4** under gcc-12.2.0 — not the gcc-14.2 paths in gpu docs).

#### Step 1 — validate tier libs (skip rebuild when OK)

`ensure_charmm_mlpot_limits.sh` selects the smallest NPR tier per cell (e.g. **8M** → `tier_8000000_nodomdec`). Jobs reuse an existing tier lib; they only rebuild when that tier’s `libcharmm.so` is missing or stale.

```bash
cd workflows/pbc_liquid_density_dyn
export MMML_CKPT=/path/to/DESdimers_params.json
bash scripts/check_pc_bach_step1.sh
```

Manual spot-check for one cell size:

```bash
source ../../scripts/pc_bach_env.sh
bash ../../scripts/check_charmm_tier_lib.sh --n-ml 2660 --pbc --box-size 32 --pc-bach
ls -l ~/.cache/mmml-charmm-build/tier_*_nodomdec/lib/libcharmm.so
```

If Step 1 fails, build tiers once (Step 2) then re-run Step 1.

#### Step 2 — build tier libs (only when Step 1 reports missing/stale)

Source the pc-bach OpenMPI stack first, then prebuild all matrix tiers:

```bash
source ../../scripts/pc_bach_env.sh
bash scripts/prebuild_charmm_tiers.sh
```

Fresh `libcharmm.so` on pc-bach may require a **PIC FFTW** built locally (module `fftw` sometimes lacks `-fPIC`). Build/install FFTW with position-independent code, then point CMake at it before `rebuild_charmm_mlpot.sh` / `ensure_charmm_mlpot_limits.sh`. Skip this entirely when Step 1 already passes.

#### Step 3 — job environment (Slurm prolog / interactive)

Add to your Slurm batch prolog or shell before `md-system` (paths replace gcc-14.2 gpu defaults):

```bash
source /path/to/mmml/scripts/pc_bach_env.sh
# module load gcc/gcc-12.2.0-cmake-3.25.1-openmpi-4.1.4   # optional if pc_bach_env.sh loads them
# module load charmm/c47a2-gcc-12.2.0-openmpi-4.1.4
export OPENMPI_ROOT=/opt/gcc-12.2.0/openmpi-4.1.4/build
export PATH="$OPENMPI_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_ROOT/lib:${LD_LIBRARY_PATH:-}"
# CHARMM_LIB_DIR set per job by ensure_charmm_mlpot_limits.sh, e.g.:
# export CHARMM_LIB_DIR=$HOME/.cache/mmml-charmm-build/tier_8000000_nodomdec/lib
export MMML_CKPT=/path/to/DESdimers_params.json
export JAX_ENABLE_X64=1
export MMML_MLPOT_DEVICE=cpu JAX_PLATFORMS=cpu

MMML_MPI_NP=1 ../../scripts/mmml-charmm-mpirun.sh md-system --config ...   # smoke
```

`job_shell.sh` and `snakemake_slurm_cpu.sh` source `pc_bach_env.sh` automatically when `scheduler: cpu` or `cluster: pc-bach` in config.

Uses `profiles/slurm-cpu` (no `--gres=gpu`) and `config.pc-bach.cpu.yaml`:

```bash
cd workflows/pbc_liquid_density_dyn
bash scripts/preflight.sh --config config.pc-bach.cpu.yaml
bash scripts/check_pc_bach_step1.sh
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
