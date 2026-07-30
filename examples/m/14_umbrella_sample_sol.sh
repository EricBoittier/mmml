#!/usr/bin/env bash
# Solvated hybrid umbrella-sample smoke (ML solute + MM solvent).
#
#   source examples/m/_env.sh
#   bash examples/m/14_umbrella_sample_sol.sh              # TIP3 (default)
#   SOLVENT=acn bash examples/m/14_umbrella_sample_sol.sh  # acetonitrile
#
# Free GPU (second card is index 1):
#   GPU=1 SOLVENT=acn bash examples/m/14_umbrella_sample_sol.sh
#
# Default: 3 windows × 1 ps (2000 × 0.5 fs). Optional:
#   NSTEPS=500 N_WINDOWS=1   # quicker compile timing probe
#   USE_DENSITY=1            # rebuild dense box if missing (default on)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" || "${MMML_EXAMPLE_DEVICE:-}" == "gpu" || "${MMML_EXAMPLE_DEVICE:-}" == "cuda" ]]; then
  export MMML_EXAMPLE_DEVICE="${MMML_EXAMPLE_DEVICE:-gpu}"
fi

# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

export PYTHONUNBUFFERED=1
if declare -F mmml_example_env_banner >/dev/null 2>&1; then
  mmml_example_env_banner
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (physical GPU index → sole device cuda:0 for this job)"
fi

SOLVENT="$(echo "${SOLVENT:-tip3}" | tr '[:upper:]' '[:lower:]')"
case "${SOLVENT}" in
  tip3|acn|dmso) ;;
  *)
    echo "FAIL: SOLVENT=${SOLVENT} (expected tip3|acn|dmso)" >&2
    exit 1
    ;;
esac

CFG="${CFG:-${ROOT}/examples/m/yaml/umbrella_nc_${SOLVENT}.yaml}"
OUT="${OUT:-${ARTIFACTS_DIR}/umbrella_nc_${SOLVENT}}"
PSF="${ARTIFACTS_DIR}/boxes/${SOLVENT}/model.psf"
PDB="${ARTIFACTS_DIR}/boxes/${SOLVENT}/model.pdb"

if [[ ! -f "${CFG}" ]]; then
  echo "FAIL: missing config ${CFG}" >&2
  exit 1
fi

if [[ ! -f "${PSF}" || ! -f "${PDB}" ]]; then
  echo "=== missing ${SOLVENT} make-box artifacts; building dense box (BOX_SIZE=30) ==="
  export BOX_SIZE="${BOX_SIZE:-30.0}"
  export USE_DENSITY="${USE_DENSITY:-1}"
  export SOLVENT_ONLY="${SOLVENT}"
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

EXTRA=()
if [[ -n "${NSTEPS:-}" ]]; then
  EXTRA+=(--nsteps "${NSTEPS}")
fi
if [[ -n "${N_WINDOWS:-}" ]]; then
  EXTRA+=(--n-windows "${N_WINDOWS}")
fi
if [[ -n "${MAX_SEED_FORCE:-}" ]]; then
  EXTRA+=(--max-seed-force "${MAX_SEED_FORCE}")
fi

echo "=== hybrid umbrella-sample: $(basename "${CFG}") (solvent=${SOLVENT}, move-with=${MOVE_WITH}) ==="
# CLI path overrides beat YAML relatives (config-dir resolution is easy to mis-count).
uv run mmml umbrella-sample \
  --config "${CFG}" \
  --from-pdb "${PDB}" \
  --from-psf "${PSF}" \
  --checkpoint "${MMML_CKPT}" \
  --output-dir "${OUT}" \
  --move-with "${MOVE_WITH}" \
  --overwrite \
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
print("PASS: hybrid umbrella-sample -> ${OUT}")
print(f"  solvent=${SOLVENT} windows={summary['n_windows']} frames={summary['n_frames']} "
      f"ml_atoms={len(summary.get('ml_atom_indices', []))}")
print(f"  cv_label={summary.get('cv_label')}")
PY
