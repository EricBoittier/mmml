#!/usr/bin/env bash
# Hourly health monitor for liquid-density Snakemake campaigns.
#
# Usage:
#   bash scripts/monitor_health.sh           # report
#   bash scripts/monitor_health.sh --react   # report + auto-remediation
#
# Install hourly cron (login node):
#   bash scripts/install_monitor_cron.sh
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
exec "${MMML_PYTHON}" "$WORKFLOW_ROOT/scripts/monitor_health.py" "$@"
