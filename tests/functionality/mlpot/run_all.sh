#!/usr/bin/env bash
# Run MLpot exploration scripts in order from repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
DIR="tests/functionality/mlpot"

echo "Repo root: $ROOT"
echo "MMML_CKPT=${MMML_CKPT:-<not set>}"

run() {
  echo ""
  echo ">>> $*"
  python "$@"
}

run "$DIR/00_check_environment.py"
run "$DIR/01_callback_vs_ase_no_charmm.py" "$@"
run "$DIR/02_mlpot_register_smoke.py" "$@"
run "$DIR/03_energy_compare.py" "$@"

echo ""
echo "Core MLpot scripts (00-03) finished OK."
echo "Optional: 04_mlpot_minimize_stub.py --run | 05_mlpot_dynamics_stub.py --run"
