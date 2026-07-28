#!/usr/bin/env bash
# Resolve MMML / CHARMM env, then run run_job.py with the remaining argv.
# Usage (from Snakemake):
#   bash scripts/job_shell.sh --job make_boxes --output-dir ABS --status ABS ...
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
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

# Ensure a CHARMM build is visible for make-box / ADUMB (PyCHARMM).
if [[ -z "${CHARMM_LIB_DIR:-}" || ! -d "${CHARMM_LIB_DIR:-}" ]]; then
  eval "$(
    "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml 500 --pbc --box-size 32 \
      2>/dev/null | grep '^export CHARMM_LIB_DIR=' || true
  )" || true
fi
if [[ -z "${CHARMM_LIB_DIR:-}" ]]; then
  for cand in \
    "$REPO_ROOT/setup/charmm/lib" \
    "$HOME/.cache/mmml-charmm-build/tier_56000000_nodomdec/lib"
  do
    if [[ -d "$cand" ]]; then
      export CHARMM_LIB_DIR="$cand"
      break
    fi
  done
fi

# Checkpoint from workflow config when unset.
if [[ -z "${MMML_CKPT:-}" || ! -e "${MMML_CKPT}" ]]; then
  _ckpt="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, checkpoint_path
cfg = load_config(Path('${CFG}'))
print(Path('${REPO_ROOT}') / checkpoint_path(cfg))
" 2>/dev/null || true)"
  if [[ -n "${_ckpt}" ]]; then
    export MMML_CKPT="$_ckpt"
  fi
fi

echo "=== nh3_ch3cl_reaction_path job_shell ===" >&2
echo "REPO_ROOT=${REPO_ROOT}" >&2
echo "PY=${PY}" >&2
echo "MMML_CKPT=${MMML_CKPT:-<unset>}" >&2
echo "CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-<unset>}" >&2
echo "MMML_WORKFLOW_CONFIG=${MMML_WORKFLOW_CONFIG}" >&2
echo "argv: $*" >&2

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" \
  --config "$CFG" \
  --repo-root "$REPO_ROOT" \
  "$@"
