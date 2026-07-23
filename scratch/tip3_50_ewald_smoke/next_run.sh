#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: tip3_50_ewald_smoke exit=2
CMD=(
  mmml
  md-system
  --output-dir
  scratch/tip3_50_ewald_smoke
  --backend
  pycharmm
  --restart-from
  scratch/tip3_50_ewald_smoke/baseline.res
  --md-stages
  mini,heat
  --no-echeck-heat
)
exec "${CMD[@]}"
