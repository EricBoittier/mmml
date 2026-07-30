#!/usr/bin/env bash
# Production solvated hybrid umbrella (ML solute + dense TIP3 MM solvent).
#
#   source examples/m/_env.sh
#   bash examples/m/14_umbrella_sample_sol_prod.sh
#
# Optional env:
#   USE_DENSITY=1   rebuild make-box at liquid density if PSF missing (default on
#                   when boxes/tip3 is absent)
#   TIMESTEP_FS=0.25 NSTEPS=80000   safer H timestep, same 20 ps / window
#   SKIP_MBAR=1     skip umbrella-mbar after sampling
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

export PYTHONUNBUFFERED=1

CFG="${CFG:-${ROOT}/examples/m/yaml/umbrella_nc_tip3_prod.yaml}"
OUT="${OUT:-${ARTIFACTS_DIR}/umbrella_nc_tip3_prod}"
PSF="${ARTIFACTS_DIR}/boxes/tip3/model.psf"
PDB="${ARTIFACTS_DIR}/boxes/tip3/model.pdb"

if [[ ! -f "${PSF}" || ! -f "${PDB}" ]]; then
  echo "=== missing TIP3 make-box; building dense box (USE_DENSITY=1, BOX_SIZE=30) ==="
  export BOX_SIZE="${BOX_SIZE:-30.0}"
  export USE_DENSITY="${USE_DENSITY:-1}"
  bash examples/m/08_make_boxes.sh
fi
if [[ ! -f "${PSF}" || ! -f "${PDB}" ]]; then
  echo "FAIL: still missing ${PSF} / ${PDB}"
  exit 1
fi

export PSF
MOVE_WITH="$(
  uv run python - <<'PY'
from pathlib import Path
import os
from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds

psf = Path(os.environ["PSF"])
atoms, _ = read_psf_atoms_and_bonds(psf)
print(",".join(str(a.index) for a in atoms if a.resname.upper() == "AMM1"))
PY
)"

EXTRA=()
if [[ -n "${TIMESTEP_FS:-}" ]]; then
  EXTRA+=(--timestep-fs "${TIMESTEP_FS}")
fi
if [[ -n "${NSTEPS:-}" ]]; then
  EXTRA+=(--nsteps "${NSTEPS}")
fi
if [[ -n "${N_WINDOWS:-}" ]]; then
  EXTRA+=(--n-windows "${N_WINDOWS}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi

echo "=== hybrid umbrella-sample PROD: $(basename "${CFG}") ==="
echo "  out=${OUT}"
echo "  move-with=${MOVE_WITH}"
if [[ -n "${NSTEPS:-}" ]]; then
  echo "  NSTEPS override=${NSTEPS}"
fi
if [[ -n "${N_WINDOWS:-}" ]]; then
  echo "  N_WINDOWS override=${N_WINDOWS}"
fi
if [[ -n "${TIMESTEP_FS:-}" ]]; then
  echo "  TIMESTEP_FS override=${TIMESTEP_FS}"
fi
uv run mmml umbrella-sample \
  --config "${CFG}" \
  --from-pdb "${PDB}" \
  --from-psf "${PSF}" \
  --checkpoint "${MMML_CKPT}" \
  --output-dir "${OUT}" \
  --move-with "${MOVE_WITH}" \
  "${EXTRA[@]}"

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
n_win = int(summary["n_windows"])
n_fr = int(summary["n_frames"])
print(f"PASS: hybrid umbrella PROD -> ${OUT}")
print(f"  windows={n_win} frames/window={n_fr} ml_atoms={len(summary.get('ml_atom_indices', []))}")
PY

if [[ "${SKIP_MBAR:-0}" != "1" ]]; then
  echo "=== umbrella-mbar ==="
  uv run mmml umbrella-mbar --run-dir "${OUT}"
fi
