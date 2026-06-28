#!/usr/bin/env bash
# Run one liquid-density PBC dynamics campaign (Snakemake / manual).
# Usage: job_shell.sh RUN_TAG
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
RUN_TAG="${1:?usage: job_shell.sh RUN_TAG}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

_cfg_raw="${MMML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG="$_cfg_raw"
else
  CFG="$WORKFLOW_ROOT/$_cfg_raw"
fi
export MMML_WORKFLOW_CONFIG="$CFG"

read -r SCHEDULER MLPOT_DEV CFG_MPI_NP MLPOT_PROF JAX_TIMERS <<<"$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, mlpot_device_name, scheduler_mode
cfg = load_config(Path('${CFG}'))
print(
    scheduler_mode(cfg),
    mlpot_device_name(cfg),
    int(cfg.get('MMML_MPI_NP', 1)),
    int(bool(cfg.get('mlpot_profile', False))),
    int(bool(cfg.get('jax_compile_timers', False))),
)
")"

export MMML_MPI_NP="${MMML_MPI_NP:-$CFG_MPI_NP}"
if [[ "$MLPOT_PROF" == "1" ]]; then
  export MMML_MLPOT_PROFILE=1
fi
if [[ "$JAX_TIMERS" == "1" ]]; then
  export MMML_JAX_COMPILE_TIMERS=1
fi

export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-$MLPOT_DEV}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-$MLPOT_DEV}"

CLUSTER="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import cluster_name, load_config
cfg = load_config(Path('${CFG}'))
print(cluster_name(cfg))
")"

if [[ "$SCHEDULER" == "cpu" || "$CLUSTER" == "pc-bach" || "${MMML_CLUSTER:-}" == "pc-bach" ]]; then
  # shellcheck source=../../../scripts/pc_bach_env.sh
  source "$REPO_ROOT/scripts/pc_bach_env.sh"
fi

if [[ "$SCHEDULER" != "cpu" ]]; then
  if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so'; then
    echo "ERROR: libOpenCL.so.1 not found on $(hostname). Use a GPU compute node." >&2
    exit 1
  fi
fi

echo "=== pbc_liquid_density_dyn: ${RUN_TAG} ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "CONFIG=${CFG}"
echo "scheduler=${SCHEDULER} MMML_MLPOT_DEVICE=${MMML_MLPOT_DEVICE} MMML_MPI_NP=${MMML_MPI_NP}"
echo "MMML_CKPT=${MMML_CKPT:-<unset>} MMML_MLPOT_PROFILE=${MMML_MLPOT_PROFILE:-0}"

read -r N_ML BOX_SIZE WARMUP_ENABLED <<<"$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms
cfg = load_config(Path('${CFG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
warm = bool(cfg.get('warmup_mlpot_jax', True))
print(estimate_ml_atoms(cell.n_monomers, solvent=cell.solvent), cell.box_size, int(warm))
")"

eval "$(
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" --pbc --box-size "$BOX_SIZE" \
    | tee /dev/stderr \
    | grep '^export '
)"

if [[ "$WARMUP_ENABLED" == "1" ]]; then
  echo "=== warmup-mlpot-jax (serial, before mpirun) ==="
  WARMUP_ARGS="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag, warmup_mlpot_argv
cfg = load_config(Path('${CFG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(' '.join(warmup_mlpot_argv(cfg, cell)))
")"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # srun PMI is incompatible with MPI-linked --do-mm warmup; compile JAX ML path only.
    WARMUP_ARGS="${WARMUP_ARGS// --do-mm/}"
    while IFS= read -r _var; do
      [[ -n "$_var" ]] && unset "$_var" 2>/dev/null || true
    done < <(env | cut -d= -f1 | grep -E '^(OMPI_|PMI_|PMIX_|MPI_LOCALRANKID$|SLURM_MPI_TYPE$)' || true)
    export OMPI_MCA_ess=singleton
    export OMPI_MCA_mpi_init_support=0
    # shellcheck disable=SC2086
    if ! "$PY" -m mmml.cli.__main__ $WARMUP_ARGS; then
      echo "ERROR: warmup-mlpot-jax failed (serial JAX under Slurm)" >&2
      exit 1
    fi
  else
    # Snakemake Slurm jobsteps export PMI/PMIX vars without a real mpirun launch.
    while IFS= read -r _var; do
      [[ -n "$_var" ]] && unset "$_var" 2>/dev/null || true
    done < <(env | cut -d= -f1 | grep -E '^(OMPI_|PMI_|PMIX_|MPI_LOCALRANKID$|SLURM_MPI_TYPE$)' || true)
    # shellcheck disable=SC2086
    if ! "$PY" -m mmml.cli.__main__ $WARMUP_ARGS; then
      echo "ERROR: warmup-mlpot-jax failed" >&2
      exit 1
    fi
  fi
fi

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" --tag "$RUN_TAG" --config "$CFG"
