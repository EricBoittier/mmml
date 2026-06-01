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

if [[ "${RUN_EXTENDED:-0}" == "1" ]]; then
  echo ""
  echo "RUN_EXTENDED=1: running minimize + dynamics stubs"
  run "$DIR/04_mlpot_minimize_stub.py" --run --save --nstep 10 "$@"
  run "$DIR/05_mlpot_dynamics_stub.py" --run --nstep 20 "$@"
  echo ""
  echo "Extended scripts (04-05) finished OK."
else
  echo "Optional extended tests:"
  echo "  RUN_EXTENDED=1 $0"
  echo "  python $DIR/04_mlpot_minimize_stub.py --run --save --nstep 10"
  echo "  python $DIR/05_mlpot_dynamics_stub.py --run --nstep 20"
fi
