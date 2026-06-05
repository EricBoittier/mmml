#!/usr/bin/env bash
# Prepare one DCM:3 trimer geometry (Snakemake helper).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
GEOM_ID="${1:?usage: prepare_geometry_shell.sh GEOM_ID}"

cd "$REPO_ROOT"
# shellcheck source=resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

# Packmol sphere placement (same requirement as mmml md-system --composition DCM:3).
"$MMML_PYTHON" - <<'PY'
from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable

print(f"packmol: {packmol_executable()}", flush=True)
PY

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"
exec "$MPIRUN" "$WORKFLOW_ROOT/scripts/prepare_geometry.py" "$GEOM_ID" \
  --config "$WORKFLOW_ROOT/config.yaml"
