#!/usr/bin/env bash
# Step 01 — CGenFF assignment.
#
# Adds cgenff_type_idx / cgenff_charge / mol_id to a raw QM dimer NPZ. Hybrid MM
# training cannot run without these three fields.
#
# Minutes on ~40k frames. Re-running is a no-op unless LJ_FORCE_PREP=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 01: prepare-mm-dataset ==="

if [[ ! -f "${LJ_DATASET}" ]]; then
  echo "ERROR: dataset not found: ${LJ_DATASET}" >&2
  echo "       Set LJ_DATASET to a PSF-ordered dimer NPZ." >&2
  exit 2
fi

if [[ -f "${LJ_ENRICHED}" && "${LJ_FORCE_PREP:-0}" != "1" ]]; then
  echo "reusing ${LJ_ENRICHED}  (LJ_FORCE_PREP=1 to redo)"
else
  uv run mmml prepare-mm-dataset \
    --data "${LJ_DATASET}" \
    --output "${LJ_ENRICHED}" \
    --num-workers "${LJ_WORKERS:-4}" \
    ${LJ_MAX_STRUCTURES:+--max-structures "${LJ_MAX_STRUCTURES}"}
fi

# Assert rather than trust: a silent prep failure that reaches training costs
# GPU hours before anyone notices.
uv run python - "${LJ_ENRICHED}" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
d = np.load(path, allow_pickle=True)
need = {"cgenff_type_idx", "cgenff_charge", "mol_id"}
missing = need - set(d.files)
if missing:
    raise SystemExit(f"ERROR: {path} is missing {sorted(missing)}")

idx = np.asarray(d["cgenff_type_idx"])
real = idx[idx >= 0]
if real.size == 0:
    raise SystemExit(f"ERROR: {path} has no assigned atoms (all padding)")
print(f"01: OK  {len(d['E'])} frames, {len(np.unique(real))} distinct CGenFF types")
PY
