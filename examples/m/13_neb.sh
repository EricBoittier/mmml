#!/usr/bin/env bash
# Vacuum NEB smoke for NH3–CH3Cl (ASE + PhysNet kl.json).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

N_IMAGES="${N_IMAGES:-11}"
MAX_STEPS="${MAX_STEPS:-80}"
FMAX="${FMAX:-0.05}"
OUT="${ARTIFACTS_DIR}/neb"

echo "=== export NEB endpoints (if missing) ==="
if [[ ! -f examples/m/neb/reag_0_opt.xyz || ! -f examples/m/neb/prod_0_opt.xyz ]]; then
  uv run python examples/m/07_export_neb_endpoints.py
fi

echo "=== NEB: ${N_IMAGES} images, max_steps=${MAX_STEPS} ==="
uv run mmml neb \
  --config "${ROOT}/examples/m/yaml/neb.yaml" \
  --output-dir "${OUT}" \
  --n-images "${N_IMAGES}" \
  --max-steps "${MAX_STEPS}" \
  --fmax "${FMAX}" \
  --overwrite

SUMMARY="${OUT}/neb_summary.json"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "FAIL: missing ${SUMMARY}"
  exit 1
fi

uv run python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${SUMMARY}").read_text())
barrier = float(data.get("barrier_kcal_mol", float("nan")))
delta = float(data.get("delta_e_product_kcal_mol", float("nan")))
if barrier != barrier or delta != delta:
    raise SystemExit(f"FAIL: non-finite barrier={barrier!r} delta_e_product={delta!r}")
if abs(delta) < 1e-3:
    raise SystemExit(f"FAIL: |delta_e_product| too small: {delta!r}")
print(f"PASS: NEB barrier={barrier:.4f} kcal/mol, ΔE(prod)={delta:.2f} -> ${OUT}")
PY
