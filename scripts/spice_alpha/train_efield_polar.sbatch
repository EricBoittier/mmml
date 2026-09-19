#!/usr/bin/env bash
# GPU efield-train + polar loss (SPICE-α, Ef=0).
#
# Smoke (default): 2 epochs, small model — prove polar MAE prints and params write.
# Full on the 256-frame extract (valid n=13): BATCH_SIZE must be <= 13
#   sbatch --time=06:00:00 --qos=rtx4090-6hours \
#     --export=ALL,MODE=full,EPOCHS=100,BATCH_SIZE=8 \
#     scripts/spice_alpha/train_efield_polar.sbatch
# Full DES after max_frames=0: BATCH_SIZE=64 is fine.
#
# Do not source scripts/scicore_env.sh for this job: it sets JAX_ENABLE_X64=1
# and EFieldPhysNet.init fails (e3x Embed float32 vs MessagePass float64).
# The train wrapper forces JAX_ENABLE_X64=0.
#
#SBATCH --job-name=spice-ef-polar
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-30min
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=artifacts/spice_ef_polar/slurm-%j.out
#SBATCH --error=artifacts/spice_ef_polar/slurm-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
mkdir -p artifacts/spice_ef_polar

if [[ -z "${MMML_PYTHON:-}" && ! -x "$ROOT/.venv/bin/python" && -x "$HOME/mmml/.venv/bin/python" ]]; then
  export MMML_PYTHON="$HOME/mmml/.venv/bin/python"
fi
if [[ -f "$ROOT/scripts/resolve_mmml_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_mmml_env.sh"
  mmml_resolve_env "$ROOT"
  export PATH="$(dirname "${MMML_PYTHON}"):$PATH"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SPLITS="${SPLITS:-$HOME/data/spicealpha/mmml_efield/splits_des_mono}"
CKPT="${CKPT:-$ROOT/ckpts/spice_ef_polar}"
MODE="${MODE:-smoke}"

if [[ "$MODE" == "full" ]]; then
  EPOCHS="${EPOCHS:-100}"
  export BATCH_SIZE="${BATCH_SIZE:-64}"
  export FEATURES="${FEATURES:-32}"
  export MAX_DEGREE="${MAX_DEGREE:-2}"
  # 256-frame run at polar_weight=1 left valid polar mae frozen (~40 Bohr³);
  # total |E| ~1e5 eV dominates. Full-monomer default raises polar loss.
  export POLAR_WEIGHT="${POLAR_WEIGHT:-100}"
else
  EPOCHS="${EPOCHS:-2}"
  export BATCH_SIZE="${BATCH_SIZE:-8}"
  export FEATURES="${FEATURES:-16}"
  export MAX_DEGREE="${MAX_DEGREE:-1}"
fi

"$ROOT/scripts/spice_alpha/train_efield_polar.sh" "$SPLITS" "$CKPT" "$EPOCHS"

echo "PASS: efield polar train finished mode=$MODE ckpt=$CKPT"
ls -1 "$CKPT"/params-*.json | tail -n 5
