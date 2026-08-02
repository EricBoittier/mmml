#!/usr/bin/env bash
# Full solvated PMF for one solvent on gpu09. Usage: run_solvated_gpu.sh [solvent]
set -u
SOLVENT="${1:-water}"
cd /mmhome/andreychev/mmml/mmml
source examples/menshutkin/_env.sh
exec /mmhome/andreychev/mmml/mmml/.venv/bin/python -u \
  examples/menshutkin/07_solvated_pmf.py --solvent "$SOLVENT" "${@:2}"
