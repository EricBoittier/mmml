#!/usr/bin/env bash
# Vacuum NEB for NH₃–CH₃Cl with examples/m/kl.json (mmml neb).
#
# Smoke (default): 11 images, fmax=0.05
# Dense band (Asparagus-style): N_IMAGES=99 bash examples/m/13_neb.sh
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${EXAMPLE_DIR}/_env.sh"

OUT_DIR="${ARTIFACTS_DIR}/neb"
N_IMAGES="${N_IMAGES:-11}"
FMAX="${FMAX:-0.05}"
MAX_STEPS="${MAX_STEPS:-}"
CLIMB="${CLIMB:-0}"

mkdir -p "${OUT_DIR}"

CMD=(
  uv run mmml neb
  --checkpoint "${MMML_CKPT}"
  --initial "${EXAMPLE_DIR}/neb/reag_0_opt.xyz"
  --final "${EXAMPLE_DIR}/neb/prod_0_opt.xyz"
  --output-dir "${OUT_DIR}"
  --n-images "${N_IMAGES}"
  --fmax "${FMAX}"
  --pair 1,2
  --pair 0,2
  --overwrite
)

if [[ "${CLIMB}" == "1" || "${CLIMB}" == "true" ]]; then
  CMD+=(--climb)
fi
if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(--max-steps "${MAX_STEPS}")
fi

echo "=== mmml neb (kl.json × NH3–CH3Cl) → ${OUT_DIR} ==="
echo "${CMD[*]}"
"${CMD[@]}"

echo
echo "Artifacts:"
ls -la "${OUT_DIR}"
echo
echo "Pass criteria: neb_summary.json has finite barrier_kcal_mol;"
echo "  neb_profile.dat columns: RC, ΔE(kcal/mol), d_N-C, d_Cl-C."
