#!/usr/bin/env bash
# CHARMM-free PET-MAD PBC smoke: 32 Å ethanol at liquid density, 300 K, 0.5 fs.
#
#   export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
#   ./examples/pet_mad_etoh_pbc/run_smoke.sh
#
# Optional: N_STEPS, ENSEMBLE (nvt|nve), MMML_METATOMIC_DEVICE, OUT_DIR, N_MOL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CKPT="${PET_MAD_CKPT:-${MMML_CKPT:-/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt}}"
OUT_DIR="${OUT_DIR:-./scratch/pet_mad_etoh_pbc/ase_smoke}"
N_STEPS="${N_STEPS:-5}"
ENSEMBLE="${ENSEMBLE:-nvt}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export MMML_METATOMIC_DEVICE="${MMML_METATOMIC_DEVICE:-cpu}"

if [[ ! -f "$CKPT" ]]; then
  echo "set PET_MAD_CKPT to a metatomic .pt (missing: $CKPT)" >&2
  exit 2
fi

cmd=(
  uv run mmml metatomic-pbc-md
  --checkpoint "$CKPT"
  --residue ETOH
  --box-size 32
  --temperature 300
  --dt-fs 0.5
  --n-steps "$N_STEPS"
  --ensemble "$ENSEMBLE"
  --output-dir "$OUT_DIR"
)
if [[ -n "${N_MOL:-}" ]]; then
  cmd+=(--n-molecules "$N_MOL")
fi
exec "${cmd[@]}"
