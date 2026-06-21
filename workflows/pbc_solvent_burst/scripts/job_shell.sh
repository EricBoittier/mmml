#!/usr/bin/env bash
# Run one PBC solvent burst campaign (called from Snakemake).
# Usage: job_shell.sh RUN_TAG
#   e.g. job_shell.sh dcm_10_t300_l32
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
RUN_TAG="${1:?usage: job_shell.sh RUN_TAG (e.g. dcm_10_t300_l32)}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so'; then
  echo "ERROR: libOpenCL.so.1 not found on this host ($(hostname))." >&2
  echo "PyCHARMM/CHARMM must run on a GPU compute node (gpu08/gpu09), not the login node." >&2
  echo "Submit via Slurm, e.g.:" >&2
  echo "  bash scripts/snakemake_slurm.sh 1 ../../artifacts/pbc_solvent_burst/${RUN_TAG}/done.txt" >&2
  echo "Or: srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 bash scripts/job_shell.sh ${RUN_TAG}" >&2
  exit 1
fi

echo "=== pbc_solvent_burst: ${RUN_TAG} ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "MMML_CKPT=${MMML_CKPT:-<unset>}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"

N_ML="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(estimate_ml_atoms(cell.n_monomers))
")"
eval "$("$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" | grep '^export ')"

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, resolve_checkpoint, cell_from_tag
cfg = load_config(Path('${WORKFLOW_ROOT}/config.yaml'))
resolve_checkpoint(str(cfg['checkpoint']))
cell = cell_from_tag(cfg, '${RUN_TAG}')
print('Preflight OK:', cfg['checkpoint'], cell, flush=True)
"

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" --tag "$RUN_TAG" \
  --config "$WORKFLOW_ROOT/config.yaml"
