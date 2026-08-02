#!/usr/bin/env bash
#SBATCH --job-name=tip3-q0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/npt_argon_water/q0_epoch2/slurm-%j.out
#SBATCH --error=artifacts/npt_argon_water/q0_epoch2/slurm-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export XLA_PYTHON_CLIENT_PREALLOCATE=false

BOX="$ROOT/artifacts/lj_scales_des_validation/boxes/tip3"
CHECKPOINT="$ROOT/artifacts/spooky_so3lr_charges/epoch-0002"
OUT="$ROOT/artifacts/npt_argon_water/q0_epoch2"
STATUS="$OUT/status-${SLURM_JOB_ID}.tsv"
mkdir -p "$OUT"

fail() {
  rc=$?
  printf 'q0_epoch2\tFAIL\t%s\t%s\n' "$rc" "$(date -Iseconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

for required in "$CHECKPOINT/_CHECKPOINT_METADATA" "$BOX/model.psf" "$BOX/model.crd"; do
  [[ -s "$required" ]] || { echo "missing required input: $required" >&2; exit 66; }
done
printf 'q0_epoch2\tRUNNING\t0\t%s\n' "$(date -Iseconds)" > "$STATUS"

"$MMML_PYTHON" -m mmml.cli.__main__ md-system \
  --backend jaxmd --setup pbc_nve \
  --composition TIP3:732 \
  --from-psf "$BOX/model.psf" --from-crd "$BOX/model.crd" \
  --skip-cluster-build --box-size 28.0 \
  --checkpoint "$CHECKPOINT" \
  --temperature 298.15 --pressure 1.0 \
  --ps "${PS:-0.25}" --dt-fs 0.25 \
  --output-dir "$OUT" \
  --include-mm --do-ml --do-ml-dimer \
  --mm-charge-mode q0 \
  --ml-batch-size 64 --seed 123 \
  --no-calculator-pre-minimize --no-charmm-pre-minimize \
  --fire-min-steps 0 --jaxmd-minimize-steps 50 --jaxmd-pbc-minimize-steps 50 \
  --no-handoff-quality-gate \
  --jax-md-skin-distance 0.5 --jax-md-update-interval 1 \
  --steps-per-recording 100 > "$OUT/md.log" 2>&1

trap - ERR
printf 'q0_epoch2\tPASS\t0\t%s\n' "$(date -Iseconds)" > "$STATUS"
