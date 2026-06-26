#!/usr/bin/env bash
# Run neighbor-list validation scripts (not pytest). Execute from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
DIR="tests/functionality/neighbor_lists"

echo "=== NL validation ladder ==="
uv run python "$DIR/00_check_nl_env.py" "$@"
uv run python "$DIR/01_reference_oracle_smoke.py" "$@"
uv run python "$DIR/02_path_parity.py" "$@"
uv run python "$DIR/03_skin_interval_audit.py" "$@"
uv run python "$DIR/03_skin_interval_audit.py" --skin 0.5 --interval 1 "$@"
uv run python "$DIR/06_extreme_pbc_nl.py" "$@"
uv run python "$DIR/04_update_mm_pairs_integration.py" --skip-charmm "$@"

echo "=== ALL NL SCRIPTS PASSED (04 skipped CHARMM unless run manually) ==="
