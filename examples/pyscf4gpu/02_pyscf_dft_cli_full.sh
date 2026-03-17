#!/usr/bin/env bash
# Example: DFT energy + gradient + hessian + harmonic + thermo via CLI
# Run from project root: bash examples/pyscf4gpu/02_pyscf_dft_cli_full.sh
# Note: Hessian/harmonic/thermo are expensive; use small molecules.

set -e
cd "$(dirname "$0")/../.."

echo "=== 02: DFT full (energy, gradient, hessian, harmonic, thermo) ==="
uv run mmml pyscf-dft --mol examples/pyscf4gpu/water.xyz \
  --energy --gradient --hessian --harmonic --thermo \
  --output examples/pyscf4gpu/out/02_results
echo "Output: examples/pyscf4gpu/out/02_results.npz and .h5"
