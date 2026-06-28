#!/usr/bin/env bash
# Run long-range Coulomb validation ladder (MIC, jax-pme, ScaFaCoS).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PY="${PYTHON:-python3}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

echo "Using: $($PY --version)"
for script in 00_check_lr_env.py 01_mic_analytic_dimer.py 02_jax_pme_madelung.py \
  03_mic_vs_jax_pme.py 04_scafacos_methods.py 05_cross_backend_summary.py; do
  echo
  echo ">>> $script"
  $PY "$script"
done

echo
echo ">>> pytest test_coulomb_backends.py"
$PY -m pytest test_coulomb_backends.py -v --tb=short
