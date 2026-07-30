#!/usr/bin/env bash
# Production solvated hybrid umbrella (ML solute + dense MM solvent).
#
#   source examples/m/_env.sh
#   bash examples/m/14_umbrella_sample_sol_prod.sh              # TIP3
#   SOLVENT=acn bash examples/m/14_umbrella_sample_sol_prod.sh  # acetonitrile
#
# Pick a free GPU (nvidia-smi lists 0, 1, …). examples/m defaults to CPU unless
# you ask for GPU — do both in one line:
#
#   GPU=1 SOLVENT=acn bash examples/m/14_umbrella_sample_sol_prod.sh
#   # equivalent: MMML_EXAMPLE_DEVICE=gpu CUDA_VISIBLE_DEVICES=1 SOLVENT=acn …
#
# Optional env:
#   USE_DENSITY=1   rebuild make-box at liquid density if PSF missing
#   TIMESTEP_FS=0.25 NSTEPS=80000   safer H timestep, same 20 ps / window
#   SKIP_MBAR=1     skip umbrella-mbar after sampling
#   OVERWRITE=1     overwrite existing output_dir
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve GPU *before* sourcing _env.sh / starting Python (JAX binds at import).
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

CFG="${CFG:-${ROOT}/examples/m/yaml/umbrella_nc_${SOLVENT}_prod.yaml}"
OUT="${OUT:-${ARTIFACTS_DIR}/umbrella_nc_${SOLVENT}_prod}"
PSF="${ARTIFACTS_DIR}/boxes/${SOLVENT}/model.psf"
PDB="${ARTIFACTS_DIR}/boxes/${SOLVENT}/model.pdb"

if [[ ! -f "${CFG}" ]]; then
  echo "FAIL: missing config ${CFG}" >&2
  echo "      (DMSO prod YAML not added yet — use tip3/acn, or set CFG=…)" >&2
  exit 1
fi

if [[ ! -f "${PSF}" || ! -f "${PDB}" ]]; then
  echo "=== missing ${SOLVENT} make-box; building dense box (USE_DENSITY=1, BOX_SIZE=30) ==="
  export BOX_SIZE="${BOX_SIZE:-30.0}"
  export USE_DENSITY="${USE_DENSITY:-1}"
  export SOLVENT_ONLY="${SOLVENT}"
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
if [[ -n "${MAX_SEED_FORCE:-}" ]]; then
  EXTRA+=(--max-seed-force "${MAX_SEED_FORCE}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi

echo "=== hybrid umbrella-sample PROD: $(basename "${CFG}") ==="
echo "  solvent=${SOLVENT}"
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
print(f"  solvent=${SOLVENT} windows={n_win} frames/window={n_fr} "
      f"ml_atoms={len(summary.get('ml_atom_indices', []))}")
print(f"  cv_label={summary.get('cv_label')}")
PY

if [[ "${SKIP_MBAR:-0}" != "1" ]]; then
  echo "=== umbrella-mbar ==="
  uv run mmml umbrella-mbar --run-dir "${OUT}"
fi
