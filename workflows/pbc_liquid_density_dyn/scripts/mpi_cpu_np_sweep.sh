#!/usr/bin/env bash
# Legacy alias — prefer mpi_spatial_cpu_np_sweep.sh for Tier-2 spatial MPI (DCM:100).
#
# This script now delegates to the spatial MPI CPU sweep. For the old DCM:32
# non-spatial sweep, set SPATIAL_MPI=0 and use dcm32_mini_mpi.yaml.tpl manually.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SPATIAL_MPI="${SPATIAL_MPI:-1}"
if [[ "$SPATIAL_MPI" == 1 ]]; then
  exec bash "$ROOT/workflows/pbc_liquid_density_dyn/scripts/mpi_spatial_cpu_np_sweep.sh" "$@"
fi
echo "ERROR: SPATIAL_MPI=0 legacy path removed; use dcm32_mini_mpi.yaml.tpl directly" >&2
exit 1
