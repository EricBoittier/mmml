#!/usr/bin/env bash
# Coarse gas → solvent φ/ψ smoke + figure (PyCHARMM node).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export MMML_CKPT="${MMML_CKPT:-examples/sppoky-epoch-0010_params.json}"
OUT="${OUT:-artifacts/tria_phi_psi_scan}"

# Use --phi=... (equals form): a bare '-180:...' is parsed as a flag by argparse.
if [[ "${SKIP_GAS:-0}" != "1" ]]; then
  uv run python scripts/scan_trialanine_phi_psi_pes.py \
    --checkpoint "$MMML_CKPT" \
    --phi=-180:180:60 --psi=-180:180:60 \
    --out "$OUT/gas" \
    --mm-sd-steps 50 --mm-abnr-steps 50 \
    --relax-steps 80
else
  echo "SKIP_GAS=1 — using existing $OUT/gas/phi_psi_pes.npz"
fi

# Solvent: one Packmol/CHARMM box, then peptide swap + CONS DIHE per grid point
# (full rebuild-per-point tends to abort silently in libcharmm after ~8 cycles).
uv run python scripts/scan_trialanine_phi_psi_solvent.py \
  --gas-npz "$OUT/gas/phi_psi_pes.npz" \
  --out "$OUT/solvent" \
  --n-waters 40 --box-side-A 28 \
  --mm-sd-steps 50 --mm-abnr-steps 50 \
  --water-only-sd-steps 30

uv run python scripts/plot_tria_phi_psi_gas_solvent.py \
  --gas-csv "$OUT/gas/phi_psi_pes.csv" \
  --solvent-csv "$OUT/solvent/phi_psi_solvent.csv" \
  -o "$OUT/figures/gas_vs_solvent.png"

echo "Figure: $OUT/figures/gas_vs_solvent.png"
