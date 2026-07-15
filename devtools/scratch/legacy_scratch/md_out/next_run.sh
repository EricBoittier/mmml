#!/usr/bin/env bash
set -euo pipefail
# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).
# prior job: md_out exit=2
CMD=(
  mmml
  md-system
  --output-dir
  scratch/md_out
  --backend
  pycharmm
  --no-echeck-heat
)
exec "${CMD[@]}"
