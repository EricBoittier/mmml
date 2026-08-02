#!/usr/bin/env bash
#SBATCH --job-name=pes-ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/npt_argon_water/checkpoint_pes_compare_20260802/slurm-%A_%a.out
#SBATCH --error=artifacts/npt_argon_water/checkpoint_pes_compare_20260802/slurm-%A_%a.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export XLA_PYTHON_CLIENT_PREALLOCATE=false

LABELS=(
  spooky_epoch3
  spooky_epoch5
  spooky_epoch10
  spooky_step294400
  spooky_charges_epoch3
  desdimers_physnet
)
CHECKPOINTS=(
  examples/sppoky-epoch-0003_params.json
  examples/sppoky-epoch-0005_params.json
  examples/sppoky-epoch-0010_params.json
  artifacts/spooky_so3lr/step-00294400_params.json
  artifacts/spooky_so3lr_charges/epoch-0003_params.json
  examples/ckpts_json/DESdimers_params.json
)

IDX="${SLURM_ARRAY_TASK_ID:?submit as an array with --array=0-5}"
LABEL="${LABELS[$IDX]}"
CHECKPOINT="$ROOT/${CHECKPOINTS[$IDX]}"
BOX="$ROOT/artifacts/lj_scales_des_validation/boxes/tip3"
OUTROOT="$ROOT/artifacts/npt_argon_water/checkpoint_pes_compare_20260802"
OUT="$OUTROOT/$LABEL"
STATUS="$OUTROOT/status-${SLURM_ARRAY_JOB_ID}_${IDX}.tsv"
mkdir -p "$OUTROOT" "$OUT"

fail() {
  rc=$?
  printf '%s\t%s\tFAIL\t%s\t%s\n' "$LABEL" "$CHECKPOINT" "$rc" "$(date -Iseconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

for required in "$CHECKPOINT" "$BOX/model.psf" "$BOX/model.crd"; do
  [[ -s "$required" ]] || { echo "missing required input: $required" >&2; exit 66; }
done
"$MMML_PYTHON" -m mmml.cli.__main__ md-system --help >/dev/null
printf '%s\t%s\tRUNNING\t0\t%s\n' "$LABEL" "$CHECKPOINT" "$(date -Iseconds)" > "$STATUS"

"$MMML_PYTHON" -m mmml.cli.__main__ md-system \
  --backend jaxmd --setup pbc_nve \
  --composition TIP3:732 \
  --from-psf "$BOX/model.psf" --from-crd "$BOX/model.crd" \
  --skip-cluster-build --box-size 28.0 \
  --checkpoint "$CHECKPOINT" \
  --temperature 298.15 --pressure 1.0 \
  --ps "${PS:-2.0}" --dt-fs 0.25 \
  --output-dir "$OUT" \
  --include-mm --do-ml --do-ml-dimer \
  --ml-batch-size "${ML_BATCH:-1024}" --seed 123 \
  --no-calculator-pre-minimize --no-charmm-pre-minimize \
  --fire-min-steps 0 --jaxmd-minimize-steps 50 --jaxmd-pbc-minimize-steps 50 \
  --no-handoff-quality-gate --mm-charge-mode fixed \
  --jax-md-skin-distance 0.5 --jax-md-update-interval 1 \
  --steps-per-recording 100 > "$OUT/md.log" 2>&1

trap - ERR
printf '%s\t%s\tPASS\t0\t%s\n' "$LABEL" "$CHECKPOINT" "$(date -Iseconds)" > "$STATUS"
