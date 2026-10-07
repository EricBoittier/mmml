#!/usr/bin/env bash
# GPU efield-train + polar loss (SPICE-α, Ef=0).
#
# Smoke (default): 2 epochs, small model — prove polar MAE prints and params write.
# Full on the 256-frame extract (valid n=13): BATCH_SIZE must be <= 13
#   sbatch --time=06:00:00 --qos=rtx4090-6hours \
#     --export=ALL,MODE=full,EPOCHS=100,BATCH_SIZE=8 \
#     scripts/spice_alpha/train_efield_polar.sbatch
# Full DES (pad=22, polar jacrev): B=64 and B=16 both died in XLA autotune.
#   --export=ALL,MODE=big,BATCH_SIZE=4,SPLITS=...,CKPT=...
# Restart a TIMEOUT run (new CKPT dir; do not overlap a RUNNING job):
#   --export=ALL,MODE=big,BATCH_SIZE=4,RESTART=.../params-best-<uuid>.json,CKPT=..._r2
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

ROOT="${KARML_REPO:-$HOME/karml}"
cd "$ROOT"
mkdir -p artifacts/spice_ef_polar

if [[ -z "${KARML_PYTHON:-}" && ! -x "$ROOT/.venv/bin/python" && -x "$HOME/karml/.venv/bin/python" ]]; then
  export KARML_PYTHON="$HOME/karml/.venv/bin/python"
fi
if [[ -f "$ROOT/scripts/resolve_karml_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_karml_env.sh"
  karml_resolve_env "$ROOT"
  export PATH="$(dirname "${KARML_PYTHON}"):$PATH"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=0
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

SPLITS="${SPLITS:-$HOME/data/spicealpha/karml_efield/splits_des_mono}"
CKPT="${CKPT:-$ROOT/ckpts/spice_ef_polar}"
MODE="${MODE:-smoke}"

if [[ "$MODE" == "big" ]]; then
  # Capacity bump that still fits polar dμ/dEf on a 4090. Cutoff stays 10 Å:
  # DES monomers are small isolated clusters; 10 Å is already all-pairs for
  # most of them. max_degree=3 roughly doubles irreps × jacrev memory.
  EPOCHS="${EPOCHS:-100}"
  export BATCH_SIZE="${BATCH_SIZE:-4}"
  export FEATURES="${FEATURES:-64}"
  export MAX_DEGREE="${MAX_DEGREE:-2}"
  export NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
  export NUM_BASIS_FUNCTIONS="${NUM_BASIS_FUNCTIONS:-32}"
  export CUTOFF="${CUTOFF:-10.0}"
  export POLAR_WEIGHT="${POLAR_WEIGHT:-100}"
  export GRADIENT_CHECKPOINT="${GRADIENT_CHECKPOINT:-1}"
elif [[ "$MODE" == "full" ]]; then
  EPOCHS="${EPOCHS:-100}"
  export BATCH_SIZE="${BATCH_SIZE:-8}"
  export FEATURES="${FEATURES:-32}"
  export MAX_DEGREE="${MAX_DEGREE:-2}"
  export NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
  export NUM_BASIS_FUNCTIONS="${NUM_BASIS_FUNCTIONS:-10}"
  export CUTOFF="${CUTOFF:-10.0}"
  # 256-frame run at polar_weight=1 left valid polar mae frozen (~40 Bohr³);
  # total |E| ~1e5 eV dominates. Full-monomer default raises polar loss.
  export POLAR_WEIGHT="${POLAR_WEIGHT:-100}"
else
  EPOCHS="${EPOCHS:-2}"
  export BATCH_SIZE="${BATCH_SIZE:-8}"
  export FEATURES="${FEATURES:-16}"
  export MAX_DEGREE="${MAX_DEGREE:-1}"
fi

# Some sciCORE nodes (seen on sgd01) expose every GPU to every job while still
# setting CUDA_VISIBLE_DEVICES=0, so concurrent jobs pile onto one card. If the
# visible device 0 is already busy, move to the least-used visible GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  n_visible="$(nvidia-smi -L | wc -l)"
  if [[ "$n_visible" -gt 1 ]]; then
    # Pick under a per-node lock and record the claim, so jobs that start at
    # the same moment (before either has allocated memory) cannot pick the same card.
    claim_dir="$HOME/.cache/gpu_guard/$(hostname)"; mkdir -p "$claim_dir"
    exec 9>"$claim_dir/.lock"; flock 9
    for f in "$claim_dir"/*.claim; do
      [[ -e "$f" ]] || continue
      squeue -h -j "$(basename "$f" .claim)" -t R 2>/dev/null | grep -q . || rm -f "$f"
    done
    claimed="$(cat "$claim_dir"/*.claim 2>/dev/null | sort -u || true)"
    pick=""
    while IFS=, read -r idx uuid used; do
      uuid="${uuid// /}"; used="${used// /}"
      grep -qx "$uuid" <<<"$claimed" && continue
      [[ "$used" -gt 1000 ]] && continue
      pick="$idx"; pick_uuid="$uuid"; break
    done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
    if [[ -n "$pick" ]]; then
      [[ "$pick" != "${CUDA_VISIBLE_DEVICES:-0}" ]] && echo "GPU guard: using GPU $pick ($pick_uuid; ${n_visible} visible)" >&2
      export CUDA_VISIBLE_DEVICES="$pick"
      echo "$pick_uuid" > "$claim_dir/${SLURM_JOB_ID}.claim"
    else
      echo "GPU guard: no unclaimed idle GPU among ${n_visible}; keeping ${CUDA_VISIBLE_DEVICES:-0}" >&2
    fi
    flock -u 9; exec 9>&-
  fi
  mkdir -p "$CKPT"
  nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=timestamp,uuid,memory.used,utilization.gpu \
    --format=csv,noheader -l 30 > "$CKPT/gpu_usage.csv" 2>/dev/null &
  GPU_LOGGER_PID=$!
  trap 'kill "$GPU_LOGGER_PID" 2>/dev/null || true; rm -f "$HOME/.cache/gpu_guard/$(hostname)/${SLURM_JOB_ID}.claim"' EXIT
fi

echo "efield-train mode=$MODE epochs=$EPOCHS B=$BATCH_SIZE F=$FEATURES L=$MAX_DEGREE T=${NUM_ITERATIONS:-2} nbasis=${NUM_BASIS_FUNCTIONS:-10} rc=${CUTOFF:-10} polar_w=${POLAR_WEIGHT:-1}"

"$ROOT/scripts/spice_alpha/train_efield_polar.sh" "$SPLITS" "$CKPT" "$EPOCHS"

echo "PASS: efield polar train finished mode=$MODE ckpt=$CKPT"
ls -1 "$CKPT"/params-*.json | tail -n 5
