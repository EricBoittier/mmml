#!/usr/bin/env bash
# Run one liquid-methane Ewald campaign cell (called from Snakemake).
# Usage: job_shell.sh RUN_TAG
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
RUN_TAG="${1:?usage: job_shell.sh RUN_TAG}"
CONFIG="${MMML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.yaml}"
if [[ "$CONFIG" != /* ]]; then
  if [[ -f "$WORKFLOW_ROOT/$CONFIG" ]]; then
    CONFIG="$WORKFLOW_ROOT/$CONFIG"
  fi
fi

cd "$REPO_ROOT"

if [[ "${HOME:-}" == /scicore/* && -r "$REPO_ROOT/scripts/scicore_env.sh" ]]; then
  # shellcheck source=../../../scripts/scicore_env.sh
  source "$REPO_ROOT/scripts/scicore_env.sh"
fi

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

# Wrapper experiments may leave MMML_NO_MPI_RERUN=1 in the submitter env; scrub
# so CLI auto-rerun under mpirun still runs for MPI-linked CHARMM.
if [[ "${MMML_FORCE_NO_MPI_RERUN:-}" != "1" ]]; then
  unset MMML_NO_MPI_RERUN || true
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so'; then
  echo "ERROR: libOpenCL.so.1 not found on this host ($(hostname))." >&2
  echo "Submit via Slurm on a GPU compute node." >&2
  exit 1
fi

echo "=== pbc_methane_ewald: ${RUN_TAG} ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "CONFIG=${CONFIG}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"

N_ML="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag, cell_ml_atoms
cfg = load_config(Path('${CONFIG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(cell_ml_atoms(cell))
")"
BOX_SIZE="$("$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag
cfg = load_config(Path('${CONFIG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(cell.box_size)
")"
eval "$(
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" --pbc --box-size "$BOX_SIZE" \
    | tee /dev/stderr \
    | grep '^export '
)"

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, resolve_checkpoint_path, cell_from_tag, validate_checkpoint
cfg = load_config(Path('${CONFIG}'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
ckpt = resolve_checkpoint_path(cell.checkpoint)
validate_checkpoint(ckpt)
print('Preflight OK:', ckpt, cell, flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" --tag "$RUN_TAG" \
  --config "$CONFIG"
