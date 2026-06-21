#!/usr/bin/env bash
# Campaign health dashboard: leg progress, DYNA energy/T from stdout.log, restart steps.
#
# Usage:
#   bash scripts/status.sh                    # summary table + results/status.csv
#   bash scripts/status.sh -v                 # include last DYNA> lines per run
#   bash scripts/status.sh --failed -v        # failed / BAD runs only
#   bash scripts/status.sh --tag dcm_10       # deep dive one cell
#   bash scripts/status.sh --plot-dir results/plots
#   bash scripts/status.sh --json results/status.json
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

exec "${MMML_PYTHON}" "$WORKFLOW_ROOT/scripts/status.py" \
  --config "$WORKFLOW_ROOT/config.yaml" \
  "$@"
