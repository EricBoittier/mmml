#!/usr/bin/env bash
# Resume MLpot heat from an on-disk CHARMM .res (skip mini / prep ladder).
#
# Usage:
#   bash scripts/resume_heat_from_res.sh TAG
#   bash scripts/resume_heat_from_res.sh TAG /path/to/heat.res
#   MMML_WORKFLOW_CONFIG=config.prep_sweep.yaml bash scripts/resume_heat_from_res.sh \
#     resilient_dcm_52_t50_l38_ht_bussi_sw_baseline
#   bash scripts/resume_heat_from_res.sh TAG --dry-run
#
# Slurm (GPU node):
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 \
#     bash scripts/resume_heat_from_res.sh resilient_dcm_52_t50_l38_ht_bussi_sw_baseline
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_TAG [RESTART.res] [--dry-run]" >&2
  exit 1
fi

RUN_TAG="$1"
shift

RESTART_FROM=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "$RESTART_FROM" && "$1" == *.res ]]; then
        RESTART_FROM="$1"
        shift
      else
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

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

if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so'; then
  echo "ERROR: libOpenCL.so.1 not found on this host ($(hostname))." >&2
  echo "Run on a GPU compute node, e.g.:" >&2
  echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 bash scripts/resume_heat_from_res.sh ${RUN_TAG}" >&2
  exit 1
fi

echo "=== dcm_density_setup_compare heat resume: ${RUN_TAG} ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKFLOW_CONFIG=${CFG}"
echo "PY=${PY}"
echo "MMML_CKPT=${MMML_CKPT:-<unset>}"

N_ML="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag, config_for_run_tag
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms
cfg = load_config(Path('${CFG}'))
cfg = config_for_run_tag(cfg, '${RUN_TAG}')
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(estimate_ml_atoms(cell.n_monomers, solvent=cell.solvent))
")"
BOX_SIZE="$("$PY" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, cell_from_tag, config_for_run_tag
cfg = load_config(Path('${CFG}'))
cfg = config_for_run_tag(cfg, '${RUN_TAG}')
cell = cell_from_tag(cfg, '${RUN_TAG}')
print(cell.box_size)
")"
eval "$(
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" --pbc --box-size "$BOX_SIZE" \
    | tee /dev/stderr \
    | grep '^export '
)"

CMD=("$PY" "$WORKFLOW_ROOT/scripts/resume_heat_from_res.py" --tag "$RUN_TAG" --config "$CFG")
if [[ -n "$RESTART_FROM" ]]; then
  CMD+=(--restart-from "$RESTART_FROM")
fi
CMD+=("${EXTRA_ARGS[@]}")

exec "${CMD[@]}"
