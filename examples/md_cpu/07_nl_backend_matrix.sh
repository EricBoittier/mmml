#!/usr/bin/env bash
# NL parity at multiple cutoffs; optional PyCHARMM when --composition is passed through.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_cpu/_env.sh"
cd "${ROOT}"

echo "=== NL matrix: synthetic dimer, cutoffs 10 / 13 / 15 Å ==="
for cutoff in 10 13 15; do
  echo "--- cutoff=${cutoff} Å ---"
  uv run python tests/functionality/neighbor_lists/05_compare_nl_backends.py \
    --cutoff "${cutoff}" \
    --backends vesin,jax_md,ase,cell_list
done

echo "=== Extreme PBC NL cases (tight box, face wrap, orthorhombic, …) ==="
uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py \
  --backends vesin,jax_md,ase,cell_list

if [[ -n "${RUN_CHARMM_NL:-}" ]]; then
  echo "=== NL matrix: CHARMM ACO:2 (requires PyCHARMM) ==="
  uv run python tests/functionality/neighbor_lists/05_compare_nl_backends.py \
    --composition ACO:2 \
    --backends vesin,jax_md,ase,pycharmm
fi

echo "=== MM NL backend env smoke (jax_md vs cell_list via MMML_MM_NL_BACKEND) ==="
for backend in auto cell_list jax_md; do
  echo "--- MMML_MM_NL_BACKEND=${backend} ---"
  MMML_MM_NL_BACKEND="${backend}" uv run python tests/functionality/neighbor_lists/02_path_parity.py
done

echo "PASS: NL backend matrix"
