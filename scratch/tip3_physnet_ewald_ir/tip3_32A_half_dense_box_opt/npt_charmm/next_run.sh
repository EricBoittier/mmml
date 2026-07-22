#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: npt_charmm exit=1
CMD=(
  mmml
  md-system
  --output-dir
  scratch/tip3_physnet_ewald_ir/tip3_32A_half_dense_box_opt/npt_charmm
  --backend
  pycharmm
  --restart-from
  scratch/tip3_physnet_ewald_ir/tip3_32A_half_dense_box_opt/npt_charmm/equi.res
  --md-stages
  heat,equi
)
exec "${CMD[@]}"
