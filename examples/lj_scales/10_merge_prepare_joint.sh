#!/usr/bin/env bash
# Step 10 — pad-merge joint RI-MP2 NPZ to 20 atoms + prepare-mm-dataset.
#
# After ORCA collect the labeled file may already be pad-20 from the geometry
# bank; re-padding is idempotent. prepare-mm-dataset re-derives CGenFF fields
# from geometry (PSF order) so training sees consistent types.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 10: pad-merge + prepare-mm-dataset (joint) ==="

MERGED="${LJ_JOINT_MERGED}"
ENRICHED="${LJ_ENRICHED}"
BASE="${LJ_JOINT_LABELED_BASE}"

# Prefer merging train+valid+test when collect wrote all three; else train only.
MERGE_INPUTS=()
for split in train valid test; do
  f="${BASE}_${split}.npz"
  [[ -f "${f}" ]] && MERGE_INPUTS+=("${f}")
done
if [[ ${#MERGE_INPUTS[@]} -eq 0 ]]; then
  if [[ -f "${LJ_JOINT_LABELED}" ]]; then
    MERGE_INPUTS=("${LJ_JOINT_LABELED}")
  else
    echo "ERROR: no labeled splits under ${BASE}_*.npz" >&2
    echo "       Run LJ_ORCA_MODE=collect bash 09_submit_orca_rimp2.sh first." >&2
    exit 2
  fi
fi

mkdir -p "$(dirname "${MERGED}")" "$(dirname "${ENRICHED}")"

echo "  merging: ${MERGE_INPUTS[*]}"
uv run python examples/lj_scales/_pad_merge_npz.py \
  "${MERGE_INPUTS[@]}" \
  --pad-to "${LJ_PAD_TO}" \
  -o "${MERGED}"

# Keep LJ_DATASET in sync for step 05 when LJ_JOINT=1.
export LJ_DATASET="${MERGED}"

uv run mmml prepare-mm-dataset \
  --data "${MERGED}" \
  --output "${ENRICHED}" \
  --num-workers "${LJ_WORKERS:-4}" \
  ${LJ_MAX_STRUCTURES:+--max-structures "${LJ_MAX_STRUCTURES}"}

uv run python - "${ENRICHED}" <<'PY'
import sys
from collections import Counter
import numpy as np

path = sys.argv[1]
d = np.load(path, allow_pickle=True)
need = {"cgenff_type_idx", "cgenff_charge", "mol_id"}
missing = need - set(d.files)
if missing:
    raise SystemExit(f"ERROR: {path} missing {sorted(missing)}")
idx = np.asarray(d["cgenff_type_idx"])
real = idx[idx >= 0]
if real.size == 0:
    raise SystemExit(f"ERROR: {path} has no assigned atoms")
# Atom-order sanity: first DCM monomer frame should be C Cl Cl H H; ACO O-first.
Z = np.asarray(d["Z"])
N = np.asarray(d["N"])
SYMBOL = {1: "H", 6: "C", 8: "O", 17: "Cl"}
if "res_name" in d.files:
    names = [str(x) for x in d["res_name"]]
    for target, expect in (("DCM", "C Cl Cl H H"), ("ACO", "O C C C H H H H H H")):
        for i, name in enumerate(names):
            if name == target:
                z = Z[i, : int(N[i])]
                order = " ".join(SYMBOL.get(int(v), str(int(v))) for v in z)
                print(f"  {target} frame {i}: {order}")
                if order != expect:
                    print(
                        f"  WARNING: expected PSF order '{expect}' — "
                        "prepare-mm-dataset should have reordered; verify types.",
                        file=sys.stderr,
                    )
                break
print(f"10: OK  {len(d['E'])} frames, {len(np.unique(real))} CGenFF types -> {path}")
PY
