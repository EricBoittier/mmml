#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: npt_charmm2 exit=2
CMD=(
  mmml
  md-system
  --output-dir
  scratch/tip3_physnet_ewald_ir/tip3_90_box_opt/npt_charmm2
  --backend
  pycharmm
  --restart-from
  scratch/tip3_physnet_ewald_ir/tip3_90_box_opt/npt_charmm2/baseline.res
  --md-stages
  equi
)
exec "${CMD[@]}"
