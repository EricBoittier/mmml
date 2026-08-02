#!/usr/bin/env bash
# One MD arm of the denser-box / dt / ensemble campaign.
# Usage: run_one.sh <tag> <box_dir> <box_A> <ensemble> <ps> <dt_fs> <x64:0|1> <seed>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"

TAG="${1:?tag}"
BOX_DIR="${2:?box_dir}"
BOX_A="${3:?box_A}"
ENSEMBLE="${4:?nvt|npt|nve}"
PS="${5:?ps}"
DT_FS="${6:?dt_fs}"
X64="${7:?0|1}"
SEED="${8:?seed}"

CKPT=artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json
SIDECAR=artifacts/lj_scales/ckpts/hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json
OUT_ROOT=artifacts/lj_scales/dense_dt_campaign
OUT="${OUT_ROOT}/${TAG}"
PSF="${BOX_DIR}/model.psf"
CRD="${BOX_DIR}/model.crd"
# fall back to mini naming if liquid-box used different names
[[ -f "$PSF" ]] || PSF="${BOX_DIR}/mini.psf"
[[ -f "$CRD" ]] || CRD="${BOX_DIR}/mini.crd"

# Lever-2 handoff (see docs/images/dense-dt-campaign/overbind_ablation/):
#   soft (default): mm_switch_on=5 → soft wells ~lit (−3…−5); contact rays still deep
#   contact:        mm_switch_on=3.5 → kills contact −30 kcal wells but soft underbinds
# Override: DDC_HANDOFF=soft|contact  or  DDC_MM_SWITCH_ON / DDC_ML_SWITCH_WIDTH
DDC_HANDOFF="${DDC_HANDOFF:-soft}"
case "${DDC_HANDOFF}" in
  soft)
    MM_SWITCH_ON="${DDC_MM_SWITCH_ON:-5.0}"
    ML_SWITCH_WIDTH="${DDC_ML_SWITCH_WIDTH:-1.5}"
    ;;
  contact)
    MM_SWITCH_ON="${DDC_MM_SWITCH_ON:-3.5}"
    ML_SWITCH_WIDTH="${DDC_ML_SWITCH_WIDTH:-1.5}"
    ;;
  *)
    echo "ERROR: DDC_HANDOFF must be soft|contact (got ${DDC_HANDOFF})" >&2
    exit 2
    ;;
esac
MM_SWITCH_WIDTH="${DDC_MM_SWITCH_WIDTH:-5.0}"

mkdir -p "$OUT"
if [[ ! -f "$PSF" || ! -f "$CRD" ]]; then
  echo "ERROR: missing PSF/CRD under $BOX_DIR" | tee "$OUT/FAIL.txt"
  ls -la "$BOX_DIR" | tee -a "$OUT/FAIL.txt"
  exit 2
fi

case "$ENSEMBLE" in
  nvt) SETUP=pbc_nvt ;;
  npt) SETUP=pbc_npt ;;
  nve) SETUP=pbc_nve ;;
  *) echo "bad ensemble $ENSEMBLE"; exit 2 ;;
esac

if [[ "$X64" == "1" ]]; then
  export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64
  ML_DTYPE=float64
else
  export JAX_ENABLE_X64=0 MMML_ML_DTYPE=float32
  ML_DTYPE=float32
fi

{
  echo "tag=$TAG"
  echo "host=$(hostname) job=${SLURM_JOB_ID:-local} $(date -Is)"
  echo "box_dir=$BOX_DIR box_A=$BOX_A ensemble=$ENSEMBLE ps=$PS dt_fs=$DT_FS x64=$X64 seed=$SEED"
  echo "psf=$PSF crd=$CRD"
  echo "handoff=${DDC_HANDOFF} mm_switch_on=${MM_SWITCH_ON} ml_switch_width=${ML_SWITCH_WIDTH} mm_switch_width=${MM_SWITCH_WIDTH}"
  echo "note=deploy handoff differs from epoch222 train (8/1.5/5); calculator will warn — intentional lever-2 probe"
  python3 - <<PY
import json
from pathlib import Path
p=Path("$BOX_DIR")/"box.json"
if p.exists():
  print("box.json:", json.dumps(json.load(open(p)), indent=2)[:800])
PY
} | tee "$OUT/run_meta.txt"

export LJ_DEVICE=gpu JAX_PLATFORMS=cuda MMML_MLPOT_DEVICE=gpu MMML_MM_NL_DEVICE=gpu PYTHONUNBUFFERED=1

# NPT/NVE from packmol+mini alone blow up on dense L24/L26 (max|F|~140–180 eV/Å,
# E_pot explosion by step 1000). Prefer continue-from a finished NVT H5.
MINI_STEPS=50
PBC_MINI_STEPS=50
EXTRA_ARGS=()
CONTINUE_FROM=""
pick_nvt_h5() {
  local cand h5
  for cand in "$@"; do
    if [[ -f "${OUT_ROOT}/${cand}/SUCCESS.flag" ]]; then
      h5="$(find "${OUT_ROOT}/${cand}" -maxdepth 1 -name '*.h5' 2>/dev/null | head -1 || true)"
      if [[ -n "${h5:-}" && -f "$h5" ]]; then
        echo "$h5"
        return 0
      fi
    fi
  done
  return 1
}
case "$TAG" in
  L24_npt_*|L24_nve_*)
    CONTINUE_FROM="$(pick_nvt_h5 L24_nvt_dt05_x64_50ps L24_nvt_dt1_f32_50ps || true)"
    ;;
  L26_npt_*|L26_nve_*)
    CONTINUE_FROM="$(pick_nvt_h5 L26_nvt_dt05_x64_50ps L26_nvt_dt1_f32_50ps || true)"
    ;;
esac
# Soft-barostat NPT probes (tag contains softbaro): slower piston for §8 density.
if [[ "$TAG" == *softbaro* && "$ENSEMBLE" == npt ]]; then
  EXTRA_ARGS+=(--nhc-barostat-tau 50000)
  echo "softbaro: --nhc-barostat-tau 50000" | tee -a "$OUT/run_meta.txt"
fi

case "$ENSEMBLE" in
  npt|nve)
    if [[ -z "${CONTINUE_FROM}" ]]; then
      # Cold start: longer mini only; still often insufficient on L24.
      MINI_STEPS=400
      PBC_MINI_STEPS=400
    else
      MINI_STEPS=100
      PBC_MINI_STEPS=100
      # Dense NVT H5s collapse within ~1 ps (ΔE_tot ≲ −50 eV by frame 1). Seed
      # from frame 0 (post-mini / first recorded), not the destroyed last frame.
      CONT_FRAME=0
      EXTRA_ARGS+=(--continue-from "${CONTINUE_FROM}" --continue-from-frame "${CONT_FRAME}")
      echo "continue_from=${CONTINUE_FROM} frame=${CONT_FRAME}" | tee -a "$OUT/run_meta.txt"
      if [[ "$ENSEMBLE" == npt ]]; then
        # Barostat + dense hybrid: give FIRE a real shot before the piston.
        EXTRA_ARGS+=(--fire-min-steps 200)
        MINI_STEPS=300
        PBC_MINI_STEPS=300
      fi
    fi
    ;;
esac
if [[ "$ENSEMBLE" == nve ]]; then
  # Dense L24 post-FIRE max|F|~180 eV/Å; effective gate is hard-capped at 15 eV/Å
  # (base×sqrt(N/N_ref)), so raising the base cannot pass — disable for this arm.
  EXTRA_ARGS+=(--nve-max-f-start-eVA 0)
fi

set +e
/usr/bin/time -f 'elapsed_s %e' -o "$OUT/wall.time" \
  uv run mmml md-system \
    --backend jaxmd --setup "$SETUP" \
    --composition DCM:120 --box-size "$BOX_A" \
    --from-psf "$PSF" --from-crd "$CRD" --no-packmol \
    --checkpoint "$CKPT" --mm-lj-scales-file "$SIDECAR" \
    --mm-nonbond-mode jax_mic --mm-charge-mode fixed \
    --mm-switch-on "$MM_SWITCH_ON" \
    --ml-switch-width "$ML_SWITCH_WIDTH" \
    --mm-switch-width "$MM_SWITCH_WIDTH" \
    --output-dir "$OUT" --temperature 300 --pressure 1.0 --seed "$SEED" \
    --ps "$PS" --dt-fs "$DT_FS" \
    --quiet \
    --no-calculator-pre-minimize \
    --jaxmd-minimize-steps "$MINI_STEPS" --jaxmd-pbc-minimize-steps "$PBC_MINI_STEPS" \
    --fire-min-steps 0 --no-charmm-pre-minimize \
    --no-handoff-quality-gate \
    --include-mm --do-ml --do-ml-dimer \
    --ml-compute-dtype "$ML_DTYPE" --mm-nl-device gpu \
    --jax-md-skin-distance 0.5 --jax-md-update-interval 40 \
    --steps-per-recording 1000 \
    --psf-angle-restraints --psf-angle-restraint-scale 1.0 \
    "${EXTRA_ARGS[@]}" \
    >"$OUT/bench.log" 2>&1
rc=$?
set -e

# Keep RESULT logging even if wall.time is missing/corrupt (set -e must not abort here).
el=$(awk '{print $NF}' "$OUT/wall.time" 2>/dev/null || echo nan)
echo "RESULT $TAG rc=$rc wall=${el}s ensemble=$ENSEMBLE dt=$DT_FS x64=$X64 box=$BOX_A" | tee -a "$OUT_ROOT/bench.log"
if [[ $rc -ne 0 ]]; then
  echo "---- tail $TAG/bench.log ----" | tee -a "$OUT_ROOT/bench.log"
  tail -80 "$OUT/bench.log" | tee -a "$OUT_ROOT/bench.log"
fi
exit $rc
