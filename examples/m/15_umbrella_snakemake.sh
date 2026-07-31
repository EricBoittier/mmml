#!/usr/bin/env bash
# Submit parallel hybrid umbrella windows via Snakemake → GPU Slurm.
#
#   SOLVENT=acn JOBS=8 bash examples/m/15_umbrella_snakemake.sh
#   SOLVENT=tip3 JOBS=8 bash examples/m/15_umbrella_snakemake.sh
#   SMOKE=1 JOBS=3 bash examples/m/15_umbrella_snakemake.sh   # 3-window tip3
#   DRY_RUN=1 bash examples/m/15_umbrella_snakemake.sh        # snakemake -n
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WF="${ROOT}/workflows/hybrid_umbrella_windows"

SOLVENT="$(echo "${SOLVENT:-acn}" | tr '[:upper:]' '[:lower:]')"
JOBS="${JOBS:-8}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  CFG="config.smoke.yaml"
elif [[ "${SOLVENT}" == "tip3" ]]; then
  CFG="config.tip3.yaml"
elif [[ "${SOLVENT}" == "acn" ]]; then
  CFG="config.yaml"
else
  echo "FAIL: SOLVENT=${SOLVENT} (use tip3|acn, or set MMML_WORKFLOW_CONFIG=...)" >&2
  exit 1
fi

export MMML_WORKFLOW_CONFIG="${MMML_WORKFLOW_CONFIG:-${CFG}}"
cd "${WF}"

EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA+=(-n)
fi

if [[ "${LOCAL:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  exec bash scripts/snakemake_local.sh "${JOBS}" "${EXTRA[@]}"
fi

echo "=== snakemake GPU: config=${MMML_WORKFLOW_CONFIG} -j${JOBS} ==="
exec bash scripts/snakemake_slurm.sh "${JOBS}" "${EXTRA[@]}"
