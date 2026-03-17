#!/usr/bin/env bash
# Example: MP2 (post-HF) energy + gradient via CLI
# Run from project root: bash examples/pyscf4gpu/03_pyscf_mp2_cli.sh

set -e
cd "$(dirname "$0")/../.."

echo "=== 03: MP2 energy + gradient (CLI) ==="
uv run mmml pyscf-mp2 --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" \
  --energy --gradient --output examples/pyscf4gpu/out/03_results
echo "Output: examples/pyscf4gpu/out/03_results.npz and .h5"
