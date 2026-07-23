#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: tip3_90_smoke exit=1
CMD=(
  mmml
  md-system
  --output-dir
  scratch/tip3_physnet_ewald_ir/tip3_90_smoke
  --backend
  pycharmm
  --restart-from
  scratch/tip3_physnet_ewald_ir/tip3_90_smoke/baseline.res
  --md-stages
  mini,heat,nve
  --no-echeck-heat
)
exec "${CMD[@]}"
