#!/usr/bin/env bash
# Run a single hybrid umbrella window (writes output_dir/windows/wXXX.npz).
set -euo pipefail

YAML=""
PSF=""
PDB=""
CRD=""
CKPT=""
OUT=""
WID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml) YAML="$2"; shift 2 ;;
    --psf) PSF="$2"; shift 2 ;;
    --pdb) PDB="$2"; shift 2 ;;
    --crd) CRD="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --output-dir) OUT="$2"; shift 2 ;;
    --window) WID="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in YAML PSF PDB CKPT OUT WID; do
  if [[ -z "${!req}" ]]; then
    echo "FAIL: --${req,,} required" >&2
    exit 2
  fi
done

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

EXTRA=(--resume --windows "${WID}")
if [[ -n "${CRD}" && -f "${CRD}" ]]; then
  # umbrella-sample accepts PDB/PSF; CRD is optional if loader supports it later
  :
fi

echo "=== hybrid window ${WID}: $(basename "${YAML}") → ${OUT}/windows/ ==="
echo "  move-with=${MOVE_WITH}"
echo "  JAX_PLATFORMS=${JAX_PLATFORMS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

uv run mmml umbrella-sample \
  --config "${YAML}" \
  --from-pdb "${PDB}" \
  --from-psf "${PSF}" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT}" \
  --move-with "${MOVE_WITH}" \
  "${EXTRA[@]}"

WID_PAD="$(printf 'w%03d.npz' "${WID}")"
test -f "${OUT}/windows/${WID_PAD}"
echo "PASS: ${OUT}/windows/${WID_PAD}"
