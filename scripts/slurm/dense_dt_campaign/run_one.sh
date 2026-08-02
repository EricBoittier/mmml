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
  python3 - <<PY
import json
from pathlib import Path
p=Path("$BOX_DIR")/"box.json"
if p.exists():
  print("box.json:", json.dumps(json.load(open(p)), indent=2)[:800])
PY
} | tee "$OUT/run_meta.txt"

export LJ_DEVICE=gpu JAX_PLATFORMS=cuda MMML_MLPOT_DEVICE=gpu MMML_MM_NL_DEVICE=gpu PYTHONUNBUFFERED=1

# NPT/NVE need a harder handoff on dense L=24 (prior fails: E blow-up / max|F| gate).
MINI_STEPS=50
PBC_MINI_STEPS=50
EXTRA_ARGS=()
case "$ENSEMBLE" in
  npt|nve)
    MINI_STEPS=400
    PBC_MINI_STEPS=400
    ;;
esac
if [[ "$ENSEMBLE" == nve ]]; then
  # Dense L24 post-FIRE forces were ~180 eV/Å with 50-step mini; size-scaled gate ~3.7.
  EXTRA_ARGS+=(--nve-max-f-start-eVA 250)
fi

set +e
/usr/bin/time -f 'elapsed_s %e' -o "$OUT/wall.time" \
  uv run mmml md-system \
    --backend jaxmd --setup "$SETUP" \
    --composition DCM:120 --box-size "$BOX_A" \
    --from-psf "$PSF" --from-crd "$CRD" --no-packmol \
    --checkpoint "$CKPT" --mm-lj-scales-file "$SIDECAR" \
    --mm-nonbond-mode jax_mic --mm-charge-mode fixed \
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

el="nan"
[[ -s "$OUT/wall.time" ]] && el=$(awk '{print $NF}' "$OUT/wall.time")
echo "RESULT $TAG rc=$rc wall=${el}s ensemble=$ENSEMBLE dt=$DT_FS x64=$X64 box=$BOX_A" | tee -a "$OUT_ROOT/bench.log"
if [[ $rc -ne 0 ]]; then
  echo "---- tail $TAG/bench.log ----" | tee -a "$OUT_ROOT/bench.log"
  tail -80 "$OUT/bench.log" | tee -a "$OUT_ROOT/bench.log"
fi
exit $rc
