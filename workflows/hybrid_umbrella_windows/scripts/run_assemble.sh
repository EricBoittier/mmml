#!/usr/bin/env bash
# Reassemble umbrella_snapshots.npz from windows/ (no MD if all windows exist).
set -euo pipefail

YAML=""
PSF=""
PDB=""
CRD=""
CKPT=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml) YAML="$2"; shift 2 ;;
    --psf) PSF="$2"; shift 2 ;;
    --pdb) PDB="$2"; shift 2 ;;
    --crd) CRD="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --output-dir) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in YAML PSF PDB CKPT OUT; do
  if [[ -z "${!req}" ]]; then
    echo "FAIL: missing required arg" >&2
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

echo "=== assemble hybrid umbrella: ${OUT} ==="
# --no-resume-failed: pack whatever finished; do not re-run NaN windows here.
uv run mmml umbrella-sample \
  --config "${YAML}" \
  --from-pdb "${PDB}" \
  --from-psf "${PSF}" \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT}" \
  --move-with "${MOVE_WITH}" \
  --resume \
  --no-resume-failed

test -f "${OUT}/umbrella_snapshots.npz"
test -f "${OUT}/umbrella_summary.json"
echo "PASS: assembled ${OUT}/umbrella_snapshots.npz"
