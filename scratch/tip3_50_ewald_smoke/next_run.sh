#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: tip3_50_ewald_smoke exit=0
CMD=(
  mmml
  md-system
  --output-dir
  scratch/tip3_50_ewald_smoke
  --backend
  pycharmm
  --md-stages
  mini,heat,nve
)
exec "${CMD[@]}"
