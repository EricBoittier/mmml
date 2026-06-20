#!/usr/bin/env bash
# Launch pbc_solvent_burst on Slurm via Snakemake executor plugin.
#
# Usage: snakemake_slurm.sh [MAX_JOBS]
#   MAX_JOBS defaults to tier pools from config.yaml (fast + slow when tiering on).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

IFS=$'\t' read -r DEFAULT_JOBS DEFAULT_RES <<EOF
$("$PY" -c "
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, slurm_launch_jobs, slurm_resources_cli
cfg = load_config()
print(f\"{slurm_launch_jobs(cfg)}\t{slurm_resources_cli(cfg)}\")
")
EOF

if [[ -z "${DEFAULT_RES:-}" ]]; then
  echo "snakemake_slurm.sh: failed to resolve --resources from config.yaml" >&2
  exit 1
fi

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

if [[ "${MMML_SNAKEMAKE_FORCE:-}" != "1" ]]; then
  _existing=()
  while IFS= read -r _pid; do
    _cwd="$(readlink -f "/proc/${_pid}/cwd" 2>/dev/null || true)"
    if [[ "$_cwd" == "$WORKFLOW_ROOT" ]]; then
      _existing+=("$_pid")
    fi
  done < <(pgrep -f 'snakemake --profile profiles/slurm|uv run --with snakemake' 2>/dev/null || true)
  if ((${#_existing[@]} > 0)); then
    echo "snakemake_slurm.sh: driver already running in ${WORKFLOW_ROOT} (PIDs: ${_existing[*]})." >&2
    echo "  bash scripts/stop_snakemake.sh" >&2
    echo "  snakemake --profile profiles/slurm --unlock" >&2
    echo "  Or force a second driver: MMML_SNAKEMAKE_FORCE=1 bash scripts/snakemake_slurm.sh ..." >&2
    exit 1
  fi
fi

echo "Snakemake Slurm: -j${JOBS} --resources ${DEFAULT_RES}" >&2

# shellcheck disable=SC2086
exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
