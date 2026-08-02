#!/bin/bash
#SBATCH --job-name=des-check
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/dimer_2d/des-check-%j.out
#SBATCH --error=artifacts/dimer_2d/des-check-%j.err
set -uo pipefail
REPO="$HOME/mmml_npt"; cd "$REPO" || exit 1
export MMML_SCICORE_CMAKE="CMake/3.31.8-GCCcore-14.3.0" MMML_SCICORE_TOOLCHAIN="foss/2025a"
source scripts/scicore_env.sh
[[ -f "$REPO/setup/charmm/libcharmm.so" ]] || { export CHARMM_LIB_DIR="$HOME/mmml/setup/charmm"; export CHARMM_HOME="$HOME/mmml/setup/charmm"; }
source .venv/bin/activate
export PYTHONPATH="$REPO" UV_NO_SYNC=1 UV_OFFLINE=1
CKPT="$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307"
SUB="$REPO/artifacts/dimer_2d/des_water300.npz"
out="$REPO/artifacts/dimer_2d/des_water"; rm -rf "$out"; mkdir -p "$out"
# Does the model reproduce its OWN training data through this evaluate path?
uv run mmml md-system --backend jaxmd --setup pbc_nve \
  --checkpoint "$CKPT/epoch-25" \
  --evaluate-npz "$SUB" --evaluate-reference-npz "$SUB" --max-frames 300 \
  --evaluate-output "$out/evaluate.json" --evaluate-forces-npz "$out/evaluate.npz" \
  --output-dir "$out" --composition TIP3:2 --include-mm --do-ml --do-ml-dimer --ml-batch-size 64 \
  > "$out/eval.log" 2>&1
echo "exit $?"
ls "$out" | head
