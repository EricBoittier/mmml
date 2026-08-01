#!/usr/bin/env bash
# Step 12 — build the DES training dataset (LJ_DES=1).
#
#   qcell_dimers.h5  ->  padded NPZ  ->  CGenFF-enriched  ->  top-N residues
#
# Runs where the HDF5 is (scicore). Each stage is skipped when its output
# already exists; LJ_FORCE_PREP=1 redoes all three.
#
# The residue cut is the point of this step. The full DES set reaches 90 CGenFF
# LJ types, but 25 of them appear in under 1,000 frames — a trainable sigma/eps
# scale on a type that thin drifts without being constrained by data. Keeping
# the best-sampled LJ_DES_TOP_RESIDUES residues holds every reachable type
# above ~1,300 frames at the cost of ~25% of the frames.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
LJ_DES=1 source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 12: DES dimer dataset (top ${LJ_DES_TOP_RESIDUES} residues) ==="

if [[ ! -f "${LJ_DES_H5}" ]]; then
  echo "ERROR: source HDF5 not found: ${LJ_DES_H5}" >&2
  echo "       It lives on scicore; set LJ_DES_H5 to point at it." >&2
  echo "       See docs/des-so3lr-dimers.md." >&2
  exit 2
fi

# --- 12a: HDF5 -> padded NPZ ------------------------------------------------
if [[ -f "${LJ_DES_RAW}" && "${LJ_FORCE_PREP:-0}" != "1" ]]; then
  echo "12a: reusing ${LJ_DES_RAW}  (LJ_FORCE_PREP=1 to redo)"
else
  uv run python scripts/des_h5_to_npz.py "${LJ_DES_H5}" \
    -o "${LJ_DES_RAW}" \
    --pad "${LJ_DES_PAD}" \
    ${LJ_DES_MAX_STRUCTURES:+--max-structures "${LJ_DES_MAX_STRUCTURES}"}
fi

# --- 12b: CGenFF assignment -------------------------------------------------
# Done once over everything; the residue cut is applied afterwards so a
# different LJ_DES_TOP_RESIDUES does not re-pay this cost.
if [[ -f "${LJ_DES_ALL}" && "${LJ_FORCE_PREP:-0}" != "1" ]]; then
  echo "12b: reusing ${LJ_DES_ALL}  (LJ_FORCE_PREP=1 to redo)"
else
  uv run mmml prepare-mm-dataset \
    --data "${LJ_DES_RAW}" \
    --output "${LJ_DES_ALL}" \
    --num-workers "${LJ_WORKERS:-8}"
fi

# --- 12c: residue cut -------------------------------------------------------
uv run python scripts/filter_mm_dataset_by_residue.py "${LJ_DES_ALL}" \
  --top "${LJ_DES_TOP_RESIDUES}" \
  -o "${LJ_ENRICHED}"

# Assert rather than trust: a silent prep failure that reaches training costs
# GPU hours before anyone notices.
uv run python - "${LJ_ENRICHED}" <<'PY'
import sys
from collections import Counter
import numpy as np

path = sys.argv[1]
d = np.load(path, allow_pickle=True)
need = {"cgenff_type_idx", "cgenff_charge", "mol_id", "cgenff_res_name"}
missing = need - set(d.files)
if missing:
    raise SystemExit(f"ERROR: {path} is missing {sorted(missing)}")

idx = np.asarray(d["cgenff_type_idx"])
real = idx[idx >= 0]
if real.size == 0:
    raise SystemExit(f"ERROR: {path} has no assigned atoms (all padding)")

res = np.asarray(d["cgenff_res_name"]).astype(str)
counts = Counter(res.ravel().tolist())
n = len(d["E"])
print(f"12: OK  {n:,} frames, {len(np.unique(real))} CGenFF types, "
      f"{len(counts)} residues")
print(f"    top residues: "
      + ", ".join(f"{r}({c:,})" for r, c in counts.most_common(8)))
# Water dominates DES; say so rather than letting it surprise someone at
# eval time when every error metric is really a water metric.
n_wat = int(np.sum(np.any(res == "TIP3", axis=1)))
print(f"    TIP3-containing frames: {n_wat:,} ({100 * n_wat / n:.1f}%)")
PY

echo
echo "Next:  LJ_DES=1 LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh"
