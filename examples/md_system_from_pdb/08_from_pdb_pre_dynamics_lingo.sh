#!/usr/bin/env bash
# Full-system PDB → PyCHARMM NVE with pre-dynamics CHARMM lingo (cons fix).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/08_from_pdb_pre_dynamics_lingo.yaml"
OUT="${ARTIFACTS_DIR}/08_pre_dynamics_lingo"
LINGO_INP="${OUT}/pycharmm_pre_dynamics_lingo.inp"

echo "=== config $(basename "${CFG}") (pre-dynamics lingo) ==="
uv run mmml md-system \
  --config "${CFG}" \
  --from-pdb "${PDB_MONOMER}" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

if [[ ! -f "${LINGO_INP}" ]]; then
  echo "FAIL: missing ${LINGO_INP}" >&2
  exit 1
fi
if ! grep -q "cons fix sele resid 1 end" "${LINGO_INP}"; then
  echo "FAIL: ${LINGO_INP} missing expected cons fix line" >&2
  exit 1
fi

echo "PASS: pre-dynamics lingo smoke -> ${OUT}"
echo "Check: ${LINGO_INP} and log line 'Pre-dynamics CHARMM lingo'"
