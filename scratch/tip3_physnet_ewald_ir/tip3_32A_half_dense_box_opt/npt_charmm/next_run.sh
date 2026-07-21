#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: npt_charmm exit=2
CMD=(
  mmml
  md-system
  --output-dir
  scratch/tip3_physnet_ewald_ir/tip3_32A_half_dense_box_opt/npt_charmm
  --backend
  pycharmm
  --restart-from
  scratch/tip3_physnet_ewald_ir/tip3_32A_half_dense_box_opt/npt_charmm/baseline.res
  --md-stages
  mini,heat,equi
)
exec "${CMD[@]}"
