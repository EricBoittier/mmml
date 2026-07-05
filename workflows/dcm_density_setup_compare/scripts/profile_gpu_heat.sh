#!/usr/bin/env bash
# Profile GPU MLpot + jax_mic heat on DCM:30 @ L=30 Å.
#
# Modes:
#   bash scripts/profile_gpu_heat.sh              # mini+heat (full campaign)
#   bash scripts/profile_gpu_heat.sh --heat-only  # heat-only from baseline.res
#
# Usage:
#   cd workflows/dcm_density_setup_compare
#   export MMML_CKPT=/path/to/params.json
#   export JAX_ENABLE_X64=1
#   bash scripts/profile_gpu_heat.sh [--heat-only] [RUN_TAG]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
CFG="${MMML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.profile.dcm30_l30.yaml}"
HEAT_ONLY=0
TAG=""
for arg in "$@"; do
  case "$arg" in
    --heat-only) HEAT_ONLY=1 ;;
    *) TAG="$arg" ;;
  esac
done
TAG="${TAG:-minimal_dcm_30_t50_l30_ht_bussi}"
LOG="${PROFILE_LOG:-$WORKFLOW_ROOT/profile_gpu_heat.log}"

export MMML_WORKFLOW_CONFIG="$CFG"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

export MMML_MLPOT_PROFILE="${MMML_MLPOT_PROFILE:-1}"
export MMML_JAX_COMPILE_TIMERS="${MMML_JAX_COMPILE_TIMERS:-1}"
export MMML_JAX_PME_PROFILE="${MMML_JAX_PME_PROFILE:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORKFLOW_ROOT/.jax_cache_profile_dcm30}"

mkdir -p "$(dirname "$LOG")" "$JAX_COMPILATION_CACHE_DIR"

echo "=== profile_gpu_heat: tag=${TAG} heat_only=${HEAT_ONLY} config=${CFG} ===" | tee "$LOG"
echo "LOG=${LOG}" | tee -a "$LOG"
echo "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR}" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<default>}" | tee -a "$LOG"

cd "$WORKFLOW_ROOT"
bash scripts/preflight.sh 2>&1 | tee -a "$LOG"

if [[ "$HEAT_ONLY" != "1" ]]; then
  echo "=== build + validate cluster (Packmol) ===" | tee -a "$LOG"
  # shellcheck source=../../../scripts/resolve_mmml_env.sh
  source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
  mmml_resolve_env "$REPO_ROOT"
  set +e
  "${MMML_PYTHON}" "$WORKFLOW_ROOT/scripts/build_validate_cluster.py" \
    --config "$CFG" --tag "$TAG" --mic-check 2>&1 | tee -a "$LOG"
  BUILD_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$BUILD_RC" != "0" ]]; then
    echo "ERROR: cluster build/validation failed (exit $BUILD_RC)" | tee -a "$LOG"
    exit "$BUILD_RC"
  fi
fi

if [[ "$HEAT_ONLY" == "1" ]]; then
  BASELINE="${HEAT_RESTART:-$REPO_ROOT/artifacts/dcm_density_setup_compare/profile_dcm30_l30/${TAG}/pycharmm_mini/baseline.res}"
  if [[ ! -f "$BASELINE" ]]; then
    echo "ERROR: heat-only mode needs baseline.res at $BASELINE (run full profile first or set HEAT_RESTART)" | tee -a "$LOG"
    exit 1
  fi
  echo "=== heat-only profile from $BASELINE ===" | tee -a "$LOG"
  set +e
  bash scripts/resume_heat_from_res.sh "$TAG" "$BASELINE" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
else
  set +e
  bash scripts/job_shell.sh "$TAG" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
fi

echo "=== profile_gpu_heat finished exit=${rc} ===" | tee -a "$LOG"
grep -E 'MLpot profile:|jax-pme profile|JAX compile|compile timer' "$LOG" | tail -60 | tee -a "${LOG%.log}.summary.txt" || true

exit "$rc"
