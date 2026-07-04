#!/usr/bin/env bash
# Run one DCM density × setup mini campaign (called from Snakemake).
# Usage: job_shell.sh [RUN_TAG]
#   Default (prep-sweep ovlp25 anchor):
#     bash scripts/job_shell.sh
#     srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 bash scripts/job_shell.sh
#   Other tags:
#     bash scripts/job_shell.sh minimal_dcm_77_t300_l32
#     MMML_WORKFLOW_CONFIG=config.prep_sweep.yaml bash scripts/job_shell.sh resilient_dcm_52_t50_l28_sw_baseline
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
DEFAULT_RUN_TAG="${MMML_DEFAULT_RUN_TAG:-resilient_dcm_52_t50_l28_ht_bussi_sw_ovlp25}"
RUN_TAG="${1:-$DEFAULT_RUN_TAG}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# Slurm profile forwards MMML_CKPT from the driver; fall back to dcm1 on cluster.
# shellcheck source=ckpt_defaults.sh
source "$WORKFLOW_ROOT/scripts/ckpt_defaults.sh"
export MMML_CKPT="${MMML_CKPT:-$(default_mmml_ckpt "$REPO_ROOT")}"

if [[ -n "${MMML_WORKFLOW_CONFIG:-}" ]]; then
  _cfg_raw="${MMML_WORKFLOW_CONFIG}"
else
  _cfg_raw="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import default_workflow_config_path
print(default_workflow_config_path(run_tag='${RUN_TAG}'))
")"
fi
if [[ "$_cfg_raw" = /* ]]; then
  CFG="${_cfg_raw}"
else
  CFG="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export MMML_WORKFLOW_CONFIG="$CFG"

if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so'; then
  echo "ERROR: libOpenCL.so.1 not found on this host ($(hostname))." >&2
  echo "PyCHARMM/CHARMM must run on a GPU compute node." >&2
  echo "Submit via Slurm, e.g.:" >&2
  echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 bash scripts/job_shell.sh" >&2
  echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 bash scripts/job_shell.sh ${RUN_TAG}" >&2
  exit 1
fi

echo "=== dcm_density_setup_compare: ${RUN_TAG} ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKFLOW_CONFIG=${CFG}"
echo "PY=${PY}"
echo "MMML_CKPT=${MMML_CKPT:-<unset>}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"

N_ML="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms
cfg = load_config(Path('${CFG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(estimate_ml_atoms(cell.n_monomers, solvent=cell.solvent))
")"
BOX_SIZE="$("$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag
cfg = load_config(Path('${CFG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(cell.box_size)
")"
eval "$(
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" --pbc --box-size "$BOX_SIZE" \
    | tee /dev/stderr \
    | grep '^export '
)"

WARMUP_ENABLED="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, warmup_mlpot_enabled
cfg = load_config(Path('${CFG}'))
print(int(warmup_mlpot_enabled(cfg)))
")"

if [[ "$WARMUP_ENABLED" == "1" ]]; then
  echo "=== warmup-mlpot-jax (serial, before CHARMM MLpot) ==="
  # Match deferred MPI path: registration/materialize compile on CPU until SD.
  _warmup_cpu="$("$PY" -c "
from mmml.interfaces.pycharmmInterface.charmm_mpi import (
    charmm_lib_links_mpi,
    defer_jax_warmup_until_after_mlpot_sd,
)
print(int(charmm_lib_links_mpi() and defer_jax_warmup_until_after_mlpot_sd()))
")"
  if [[ "$_warmup_cpu" == "1" ]]; then
    export MMML_MLPOT_DEVICE=cpu
    export JAX_PLATFORMS=cpu
    echo "warmup: MMML_MLPOT_DEVICE=cpu JAX_PLATFORMS=cpu (MPI defer path cache)"
  fi
  # Slurm/srun exports PMI env; ML-only warmup must not MPI_Init libcharmm at import time.
  while IFS= read -r _var; do
    [[ -n "$_var" ]] && unset "$_var" 2>/dev/null || true
  done < <(env | cut -d= -f1 | grep -E '^(OMPI_|PMI_|PMIX_|MPI_LOCALRANKID$|SLURM_MPI_TYPE$)' || true)
  export MMML_WARMUP_MLPOT_JAX_ONLY=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  _jax_threads="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config
cfg = load_config(Path('${CFG}'))
print(int(cfg.get('jax_compile_threads', cfg.get('warmup_compile_threads', 4))))
")"
  export MMML_JAX_COMPILE_THREADS="${MMML_JAX_COMPILE_THREADS:-$_jax_threads}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_jax_threads}"
  export OMPI_MCA_ess=singleton
  export OMPI_MCA_mpi_init_support=0
  export OMPI_MCA_plm=^slurm
  WARMUP_ARGS="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag, warmup_mlpot_argv
cfg = load_config(Path('${CFG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(' '.join(warmup_mlpot_argv(cfg, cell)))
")"
  # shellcheck disable=SC2086
  if ! "$PY" -m mmml.cli.__main__ $WARMUP_ARGS; then
    echo "ERROR: warmup-mlpot-jax failed" >&2
    exit 1
  fi
  unset MMML_WARMUP_MLPOT_JAX_ONLY
fi

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, resolve_checkpoint, cell_from_tag
cfg = load_config(Path('${CFG}'))
resolve_checkpoint(str(cfg['checkpoint']))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print('Preflight OK:', resolve_checkpoint(str(cfg['checkpoint'])), cell, flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" --tag "$RUN_TAG" \
  --config "$CFG"
