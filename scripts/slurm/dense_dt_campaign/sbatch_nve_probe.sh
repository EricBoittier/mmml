#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/mmhome/boittier/home/mmml}"
cd "$ROOT"
source examples/lj_scales/_env.sh
TAG="${CAMPAIGN_TAG:?}"
BOX_DIR="${CAMPAIGN_BOX_DIR:?}"
BOX_A="${CAMPAIGN_BOX_A:?}"
PS="${CAMPAIGN_PS:?}"
DT_FS="${CAMPAIGN_DT_FS:?}"
X64="${CAMPAIGN_X64:?}"
SEED="${CAMPAIGN_SEED:?}"
CKPT=artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json
SIDECAR=artifacts/lj_scales/ckpts/hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json
OUT="artifacts/lj_scales/dense_dt_campaign/${TAG}"
PSF="${BOX_DIR}/model.psf"; CRD="${BOX_DIR}/model.crd"
[[ -f "$PSF" ]] || PSF="${BOX_DIR}/mini.psf"
[[ -f "$CRD" ]] || CRD="${BOX_DIR}/mini.crd"
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
if [[ "$X64" == "1" ]]; then export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64; ML=float64
else export JAX_ENABLE_X64=0 MMML_ML_DTYPE=float32; ML=float32; fi
export LJ_DEVICE=gpu JAX_PLATFORMS=cuda MMML_MLPOT_DEVICE=gpu MMML_MM_NL_DEVICE=gpu PYTHONUNBUFFERED=1
{
  echo "tag=$TAG host=$(hostname) job=${SLURM_JOB_ID:-} $(date -Is)"
  echo "NVE probe CRD+FIRE+no-rescue dt=$DT_FS x64=$X64 ps=$PS box=$BOX_A"
} | tee "$OUT/run_meta.txt"
set +e
/usr/bin/time -f 'elapsed_s %e' -o "$OUT/wall.time" \
  uv run mmml md-system \
    --backend jaxmd --setup pbc_nve \
    --composition DCM:120 --box-size "$BOX_A" \
    --from-psf "$PSF" --from-crd "$CRD" --no-packmol \
    --checkpoint "$CKPT" --mm-lj-scales-file "$SIDECAR" \
    --mm-nonbond-mode jax_mic --mm-charge-mode fixed \
    --output-dir "$OUT" --temperature 300 --pressure 1.0 --seed "$SEED" \
    --ps "$PS" --dt-fs "$DT_FS" --quiet \
    --no-calculator-pre-minimize \
    --jaxmd-minimize-steps 400 --jaxmd-pbc-minimize-steps 400 \
    --fire-min-steps 300 --no-charmm-pre-minimize \
    --no-handoff-quality-gate \
    --include-mm --do-ml --do-ml-dimer \
    --ml-compute-dtype "$ML" --mm-nl-device gpu \
    --jax-md-skin-distance 0.5 --jax-md-update-interval 40 \
    --steps-per-recording 500 \
    --psf-angle-restraints --psf-angle-restraint-scale 1.0 \
    --nve-max-f-start-eVA 0 \
    --nve-etot-drift-abort-eV 0.5 \
    --no-nve-etot-drift-rescue \
    >"$OUT/bench.log" 2>&1
rc=$?
set -e
el=$(awk '{print $NF}' "$OUT/wall.time" 2>/dev/null || echo nan)
echo "RESULT $TAG rc=$rc wall=${el}s ensemble=nve dt=$DT_FS x64=$X64 box=$BOX_A" | tee -a artifacts/lj_scales/dense_dt_campaign/bench.log
[[ $rc -ne 0 ]] && tail -80 "$OUT/bench.log" | tee -a artifacts/lj_scales/dense_dt_campaign/bench.log
[[ $rc -eq 0 ]] && touch "$OUT/SUCCESS.flag"
exit $rc
