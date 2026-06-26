#!/usr/bin/env bash
# Neighbor-list backend parity (synthetic cluster; no CHARMM).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_cpu/_env.sh"
cd "${ROOT}"

echo "=== NL parity (vesin / jax-md / ase / cell_list) ==="
uv run python tests/functionality/neighbor_lists/05_compare_nl_backends.py \
  --backends vesin,jax_md,ase,cell_list \
  "$@"
