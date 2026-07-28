#!/usr/bin/env bash
# Solvated hybrid umbrella-sample smoke (ML solute + MM TIP3 solvent).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

CFG="${CFG:-${ROOT}/examples/m/yaml/umbrella_nc_tip3.yaml}"
OUT="${ARTIFACTS_DIR}/umbrella_nc_tip3"
PSF="${ARTIFACTS_DIR}/boxes/tip3/model.psf"
PDB="${ARTIFACTS_DIR}/boxes/tip3/model.pdb"

if [[ ! -f "${PSF}" || ! -f "${PDB}" ]]; then
  echo "=== missing TIP3 make-box artifacts; building (BOX_SIZE=30) ==="
  export BOX_SIZE="${BOX_SIZE:-30.0}"
  bash examples/m/08_make_boxes.sh
fi
if [[ ! -f "${PSF}" || ! -f "${PDB}" ]]; then
  echo "FAIL: still missing ${PSF} / ${PDB}"
  exit 1
fi

# Resolve NH3 move-with indices (N1 + H*) from the PSF for stretch seeding.
export PSF
MOVE_WITH="$(
  uv run python - <<'PY'
from pathlib import Path
import os
from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds

psf = Path(os.environ["PSF"])
atoms, _ = read_psf_atoms_and_bonds(psf)
idxs = []
for a in atoms:
    if a.resname.upper() != "AMM1":
        continue
    idxs.append(str(a.index))
print(",".join(idxs))
PY
)"

echo "=== hybrid umbrella-sample: $(basename "${CFG}") (move-with=${MOVE_WITH}) ==="
uv run mmml umbrella-sample \
  --config "${CFG}" \
  --output-dir "${OUT}" \
  --move-with "${MOVE_WITH}" \
  --overwrite

SUMMARY="${OUT}/umbrella_summary.json"
SNAP="${OUT}/umbrella_snapshots.npz"
for f in "${SUMMARY}" "${SNAP}"; do
  if [[ ! -f "${f}" ]]; then
    echo "FAIL: missing ${f}"
    exit 1
  fi
done

uv run python - <<PY
import json
from pathlib import Path
import numpy as np
summary = json.loads(Path("${SUMMARY}").read_text())
assert summary.get("engine") == "hybrid_jaxmd", summary.get("engine")
snap = np.load("${SNAP}", allow_pickle=True)
assert "energies_unbiased_ev" in snap.files, snap.files
assert "ml_atom_indices" in snap.files, snap.files
print("PASS: hybrid umbrella-sample -> ${OUT}")
print(f"  windows={summary['n_windows']} frames={summary['n_frames']} "
      f"ml_atoms={len(summary.get('ml_atom_indices', []))}")
PY
