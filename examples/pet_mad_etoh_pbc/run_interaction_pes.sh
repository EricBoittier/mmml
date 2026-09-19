#!/usr/bin/env bash
# CHARMM-free PET-MAD interaction slices / surface / trimer leftover.
#
#   export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
#   ./examples/pet_mad_etoh_pbc/run_interaction_pes.sh
#
# Replot without PET: FROM_JSON=examples/pet_mad_etoh_pbc/data/interaction_pes.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CKPT="${PET_MAD_CKPT:-${MMML_CKPT:-/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt}}"
OUT_DIR="${OUT_DIR:-./scratch/pet_interaction_pes}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export MMML_METATOMIC_DEVICE="${MMML_METATOMIC_DEVICE:-cpu}"

cmd=(uv run mmml pet-interaction-pes --output-dir "$OUT_DIR")
if [[ -n "${FROM_JSON:-}" ]]; then
  cmd+=(--from-json "$FROM_JSON")
else
  if [[ ! -f "$CKPT" ]]; then
    echo "set PET_MAD_CKPT to a metatomic .pt (missing: $CKPT)" >&2
    exit 2
  fi
  cmd+=(--checkpoint "$CKPT")
fi
exec "${cmd[@]}"
