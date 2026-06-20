#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
exec python3 "$WORKFLOW_ROOT/scripts/status.py" "$@"
