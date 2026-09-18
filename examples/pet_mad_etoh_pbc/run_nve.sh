#!/usr/bin/env bash
# PET-MAD NVE conservation on 32 Å liquid ethanol (FIRE mini, then VelocityVerlet).
#
#   export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
#   ./examples/pet_mad_etoh_pbc/run_nve.sh
#
# Optional: N_STEPS (default 400 → 0.2 ps at 0.5 fs), MINI_STEPS, MINI_FMAX,
# MMML_METATOMIC_DEVICE, OUT_DIR, N_MOL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CKPT="${PET_MAD_CKPT:-${MMML_CKPT:-/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt}}"
OUT_DIR="${OUT_DIR:-./scratch/pet_mad_etoh_pbc/ase_nve}"
N_STEPS="${N_STEPS:-400}"
MINI_STEPS="${MINI_STEPS:-60}"
MINI_FMAX="${MINI_FMAX:-0.2}"
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
  --ensemble nve
  --minimize-steps "$MINI_STEPS"
  --minimize-fmax "$MINI_FMAX"
  --n-steps "$N_STEPS"
  --log-every 1
  --output-dir "$OUT_DIR"
)
if [[ -n "${N_MOL:-}" ]]; then
  cmd+=(--n-molecules "$N_MOL")
fi
exec "${cmd[@]}"
