#!/usr/bin/env bash
# 30 Å TIP3 box: export solute → make-box → md-system from PDB.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

export BOX_SIZE="${BOX_SIZE:-30.0}"
export USE_DENSITY="${USE_DENSITY:-1}"

echo "=== solute PDB (BOX_SIZE=${BOX_SIZE}) ==="
uv run python examples/m/07_export_solute_pdb.py

has_pycharmm=0
if uv run python -c "import pycharmm" >/dev/null 2>&1; then
  has_pycharmm=1
fi
if [[ "${has_pycharmm}" != "1" ]]; then
  echo "SKIP: PyCHARMM not importable — cannot build make-box"
  exit 0
fi

echo "=== make-box TIP3 (L=${BOX_SIZE} Å, USE_DENSITY=${USE_DENSITY}) ==="
bash examples/m/08_make_boxes.sh

pdb="${ARTIFACTS_DIR}/boxes/tip3/model.pdb"
psf="${ARTIFACTS_DIR}/boxes/tip3/model.psf"
if [[ ! -f "${pdb}" ]]; then
  echo "FAIL: missing ${pdb}"
  exit 1
fi
if [[ ! -f "${psf}" ]]; then
  echo "FAIL: missing ${psf} (make-box must write model.psf; jaxmd-unified"
  echo "      loads the sibling PSF — regenerating a TIP3 box via sequence_string"
  echo "      overflows CHARMM and aborts with ABNORMAL TERMINATION LEVEL 0)"
  exit 1
fi
if [[ -z "${MMML_CGENFF_EXTRA_RTF:-}" ]]; then
  echo "WARN: MMML_CGENFF_EXTRA_RTF unset — source examples/m/_env.sh for CH3CL"
fi

echo "=== md-system sol_tip3_30A (jaxmd unified) ==="
uv run mmml md-system --config examples/m/yaml/sol_tip3_30A_md.yaml

echo "PASS: solvated TIP3 30 Å -> ${ARTIFACTS_DIR}/sol_tip3_30A"
