#!/usr/bin/env bash
# Step 08 — joint ACO+DCM geometry bank (thermal NMS + dimer grid + heteros).
#
# Builds exhaustive intermolecular coverage (directions × orientations × r_com)
# with independent thermal normal-mode conformers per monomer. Does NOT run
# ORCA — labels come from step 09.
#
# Requires a source NPZ that already carries res_name + CGenFF fields (default:
# examples/mp2_nms15_train.npz). Geometry-only output is the ORCA input bank.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 08: joint ACO+DCM geometry bank (NMS) ==="

if [[ "${LJ_NMS_CONFORMERS}" -lt 2 ]]; then
  echo "ERROR: LJ_NMS_CONFORMERS must be >= 2 (got ${LJ_NMS_CONFORMERS})." >&2
  echo "       Thermal NMS is required; isotropic noise is not a substitute." >&2
  exit 2
fi

SRC="${LJ_GEOM_SOURCE}"
if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: geometry source not found: ${SRC}" >&2
  echo "       Need an NPZ with res_name + cgenff_* for DCM and ACO monomers." >&2
  exit 2
fi

OUT="${LJ_JOINT_GEOMS}"
mkdir -p "$(dirname "${OUT}")"

echo "  source     : ${SRC}"
echo "  nms        : conformers=${LJ_NMS_CONFORMERS} T=${LJ_NMS_TEMPERATURE} K"
echo "               freq_min=${LJ_NMS_FREQ_MIN} cm^-1"
echo "  grid       : dirs=${LJ_N_DIRECTIONS} orients=${LJ_N_ORIENTATIONS} n_r=${LJ_N_R}"
echo "  out        : ${OUT}"

uv run python scripts/make_dimer_scan_dataset.py \
  --data "${SRC}" \
  --resids DCM,ACO \
  --include-hetero \
  --include-monomers \
  --geometry-only \
  --monomer-conformers "${LJ_NMS_CONFORMERS}" \
  --nms-temperature "${LJ_NMS_TEMPERATURE}" \
  --nms-freq-min "${LJ_NMS_FREQ_MIN}" \
  --n-directions "${LJ_N_DIRECTIONS}" \
  --n-orientations "${LJ_N_ORIENTATIONS}" \
  --n-r "${LJ_N_R}" \
  --r-min "${LJ_R_MIN}" \
  --r-max "${LJ_R_MAX}" \
  --r-dense-to "${LJ_R_DENSE_TO}" \
  --seed "${LJ_GEOM_SEED}" \
  --out "${OUT}"

# Fail loudly if heteros or NMS monomers are missing.
uv run python - "${OUT}" <<'PY'
import sys
from collections import Counter
import numpy as np

path = sys.argv[1]
d = np.load(path, allow_pickle=True)
names = [str(x) for x in d["res_name"]]
c = Counter(names)
print("  composition:", dict(c))
need = {"DCM", "ACO", "DCM,DCM", "ACO,ACO", "ACO,DCM", "DCM,ACO"}
present = set(c)
# Hetero name is ACO,DCM (sorted by resid list order in the builder: DCM then ACO → DCM,ACO? )
# Builder loops i, ra then rb in resids[i+1:], with resids DCM,ACO → hetero "DCM,ACO".
if "DCM,ACO" not in present and "ACO,DCM" not in present:
    raise SystemExit("ERROR: no ACO–DCM hetero frames — check --include-hetero")
if "DCM" not in present or "ACO" not in present:
    raise SystemExit("ERROR: monomer NMS frames missing — check --include-monomers")
if "DCM,DCM" not in present or "ACO,ACO" not in present:
    raise SystemExit("ERROR: homodimer frames missing")
print(f"08: OK  {len(names)} geometries -> {path}")
PY
