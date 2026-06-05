#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "${1:?usage: collect_shell.sh WORKFLOW_ROOT CSV_PATH MD_PATH}" && pwd)"
CSV_ARG="${2:?usage: collect_shell.sh WORKFLOW_ROOT CSV_PATH MD_PATH}"
MD_ARG="${3:?usage: collect_shell.sh WORKFLOW_ROOT CSV_PATH MD_PATH}"

_resolve_out() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$WORKFLOW_ROOT" "$path"
  fi
}
CSV_PATH="$(_resolve_out "$CSV_ARG")"
MD_PATH="$(_resolve_out "$MD_ARG")"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"
exec "$PY" "$WORKFLOW_ROOT/scripts/collect_scaling.py" \
  --config "$WORKFLOW_ROOT/config.yaml" \
  --csv "$CSV_PATH" \
  --md "$MD_PATH"
